import numpy as np
import torch
from utils.paths import training_data_path
from core.analytic.shift import _apply_shift,_load_shift
from core import reduced_order_models as rom


def g_project(A: np.ndarray, G: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Project U onto span(A) with respect to the G-inner product, in full space."""
    # Allow A to come in transposed form.
    if A.shape[0] != G.shape[0] and A.shape[1] == G.shape[0]:
        A = A.T
    if U.ndim == 1:
        y = A.T @ (G @ U)              # [N]
        return A @ y                   # [N_h]
    Y = np.einsum('ih,hj,tj->ti', A.T, G, U, optimize=True)  # [T, N]
    return np.einsum('hi,ti->th', A, Y, optimize=True)       # [T, N_h]


def g_project_reduced(A: np.ndarray, G: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Reduced coordinates Y = A^T G U (timewise)."""
    if A.shape[0] != G.shape[0] and A.shape[1] == G.shape[0]:
        A = A.T
    if U.ndim == 1:
        return A.T @ (G @ U)
    return np.einsum('ih,hj,tj->ti', A.T, G, U, optimize=True)


def _g_sq_timeseries(U: np.ndarray, G: np.ndarray) -> np.ndarray:
    vals = np.einsum('ti,ij,tj->t', U, G, U, optimize=True)
    return np.maximum(vals, 0.0)


def _euclid_sq_timeseries(U: np.ndarray) -> np.ndarray:
    vals = np.einsum('ti,ti->t', U, U, optimize=True)
    return np.maximum(vals, 0.0)


def _stats_from_tag(example_name: str, reduction_tag: str):
    stats = np.load(training_data_path(example_name) / f"normalization_{reduction_tag}_{example_name}.npz")
    sol_min = stats['sol_min'].astype(np.float32)
    sol_max = stats['sol_max'].astype(np.float32)
    return sol_min, sol_max


def _normalize_reduced(Y: np.ndarray, sol_min: np.ndarray, sol_max: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return (Y - sol_min[None, :]) / (sol_max[None, :] - sol_min[None, :] + eps)


def _denormalize_reduced(Y_norm: np.ndarray, example_name: str, reduction_tag: str, device: str) -> np.ndarray:
    return rom.denormalize_solution(
        torch.tensor(Y_norm, dtype=torch.float32, device=device),
        example_name, reduction_tag=reduction_tag
    ).cpu().numpy()


def _dod_projector(
    Y_norm: np.ndarray,
    mu: np.ndarray,
    innerDOD_model,
    example_name: str,
    device: str
):
    """
    Project normalized reduced coords Y_norm[t, :] onto the dynamic N' subspace
    spanned by innerDOD_model(mu, t). Returns coefficients alpha and projected Y.
    """
    T = Y_norm.shape[0]
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    if mu_t.dim() == 0:
        mu_t = mu_t.unsqueeze(0)
    if mu_t.dim() == 1:
        mu_t = mu_t.unsqueeze(0)

    mu_norm = rom.normalize_mu(mu_t, example_name, 'N_A_reduced')  # [1, p]
    denom = float(T) if T > 0 else 1.0
    t_vals = torch.arange(T, device=device, dtype=torch.float32).view(-1, 1) / denom
    mu_batch = mu_norm.expand(T, -1)

    with torch.no_grad():
        V = innerDOD_model(mu_batch, t_vals)  # [T, N_A, N'] or [N_A, N'] if T==1
    if V.dim() == 2:
        V = V.unsqueeze(0)

    Y = torch.tensor(Y_norm, dtype=torch.float32, device=device)
    alpha = torch.einsum('tij,tj->ti', V.transpose(1, 2), Y)   # [T, N']
    Y_proj = torch.einsum('tij,tj->ti', V, alpha)              # [T, N_A]
    return alpha.detach().cpu().numpy(), Y_proj.detach().cpu().numpy()


def projector_check(
    samples,
    G: np.ndarray,
    example_name: str,
    device: str,
    kind: str,
    innerDOD_model=None,
    use_shift: bool = True,
):
    """
    Projection/normalization consistency checks.

    kind:
      - 'A_P'  : POD ambient (N_reduced) check in full space
      - 'A'    : DOD ambient (N_A_reduced) check in full space
      - 'DOD'  : Dynamic subspace (N') projection loss in reduced space
    """
    kind = kind.strip()
    if kind == 'V_mu,t':
        kind = 'DOD'

    ut0 = _load_shift(example_name) if use_shift else None

    if kind == 'A_P':
        A = np.load(training_data_path(example_name) / f"N_ambient_{example_name}.npz")["ambient"].astype(np.float32)
        reduction_tag = 'N_reduced'
        sol_min, sol_max = _stats_from_tag(example_name, reduction_tag)
        msg = "A_P^T G -> norm -> denorm -> A_P (shifted)"
        rel_err = []
        abs_err = []
        for mu, nu, sol in samples:
            U = _apply_shift(sol.astype(np.float32), ut0)
            Y = g_project_reduced(A, G, U)  # [T, N]
            Y_norm = _normalize_reduced(Y, sol_min, sol_max)
            Y_den = _denormalize_reduced(Y_norm, example_name, reduction_tag, device)
            U_hat = np.einsum('hi,ti->th', A, Y_den, optimize=True)
            err_sq = _g_sq_timeseries(U - U_hat, G)
            ref_sq = _g_sq_timeseries(U, G)
            abs_err.append(float(np.sqrt(err_sq.mean())))
            rel_err.append(float(np.sqrt(err_sq.sum()) / (np.sqrt(ref_sq.sum()) + 1e-24)))
        print(f"[{msg}]")
        print(f"Absolute Error: {np.mean(abs_err):.3e}  Relative Error: {np.mean(rel_err):.3e}")
        print("----------------------------------------")
        return

    if kind == 'A':
        A = np.load(training_data_path(example_name) / f"N_A_ambient_{example_name}.npz")["ambient"].astype(np.float32)
        reduction_tag = 'N_A_reduced'
        sol_min, sol_max = _stats_from_tag(example_name, reduction_tag)
        msg = "A^T G -> norm -> denorm -> A (shifted)"
        rel_err = []
        abs_err = []
        for mu, nu, sol in samples:
            U = _apply_shift(sol.astype(np.float32), ut0)
            Y = g_project_reduced(A, G, U)  # [T, N_A]
            Y_norm = _normalize_reduced(Y, sol_min, sol_max)
            Y_den = _denormalize_reduced(Y_norm, example_name, reduction_tag, device)
            U_hat = np.einsum('hi,ti->th', A, Y_den, optimize=True)
            err_sq = _g_sq_timeseries(U - U_hat, G)
            ref_sq = _g_sq_timeseries(U, G)
            abs_err.append(float(np.sqrt(err_sq.mean())))
            rel_err.append(float(np.sqrt(err_sq.sum()) / (np.sqrt(ref_sq.sum()) + 1e-24)))
        print(f"[{msg}]")
        print(f"Absolute Error: {np.mean(abs_err):.3e}  Relative Error: {np.mean(rel_err):.3e}")
        print("----------------------------------------")
        return

    if kind == 'DOD':
        if innerDOD_model is None:
            raise ValueError("innerDOD_model must be provided for kind='DOD'.")
        A = np.load(training_data_path(example_name) / f"N_A_ambient_{example_name}.npz")["ambient"].astype(np.float32)
        reduction_tag = 'N_A_reduced'
        sol_min, sol_max = _stats_from_tag(example_name, reduction_tag)

        rel_err = []
        abs_err = []
        for mu, nu, sol in samples:
            U = _apply_shift(sol.astype(np.float32), ut0)
            Y = g_project_reduced(A, G, U)          # [T, N_A]
            Y_norm = _normalize_reduced(Y, sol_min, sol_max)

            _, Y_proj_norm = _dod_projector(Y_norm, np.asarray(mu), innerDOD_model, example_name, device)
            Y_proj = _denormalize_reduced(Y_proj_norm, example_name, reduction_tag, device)
            U_hat = np.einsum('hi,ti->th', A, Y_proj, optimize=True)

            err_sq = _g_sq_timeseries(U - U_hat, G)
            ref_sq = _g_sq_timeseries(U, G)
            abs_err.append(float(np.sqrt(err_sq.mean())))
            rel_err.append(float(np.sqrt(err_sq.sum()) / (np.sqrt(ref_sq.sum()) + 1e-24)))

        print("[A^T G -> norm -> inner^T -> inner -> denorm -> A (shifted)]")
        print(f"Absolute Error: {np.mean(abs_err):.3e}  Relative Error: {np.mean(rel_err):.3e}")
        print("----------------------------------------")
        return

    raise ValueError(f"Unknown projector_check kind: {kind}")
