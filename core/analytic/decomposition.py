import numpy as np
import torch
from utils.paths import training_data_path
from core.analytic.norms import g_norm_2
from core.analytic.project import g_project

def _ensure_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# The following two methods rely on the implementation mentioned
# in the paper Error Estimates for POD-DL-ROM by Brivio

def pod_dl_error_decomposition(
    fw_pod_fn,                    
    samples,
    example_name: str,
    G: np.ndarray
):
    """
      E_R, E_S, E_POD, E_POD_inf, E_NN and bounds.
    """
    A_P = np.load(training_data_path(example_name) / f'N_ambient_{example_name}.npz')['ambient'].astype(np.float32)
    train_blob = np.load(training_data_path(example_name) / f'full_order_training_data_{example_name}.npz') 
    S_train_all = train_blob['solution'].astype(np.float32)

    # Unexplained variance on TRAIN via G-orth projection onto A_P
    uv_train = []
    for U in S_train_all:                                  # U: [T, N_h]
        U_proj = g_project(A_P, G, U)              # [T, N_h]
        d_sq = g_norm_2(U - U_proj, G)             # [T]
        uv_train.append(float(d_sq.mean()))
    unexplained_var_train = float(np.mean(uv_train))        # scalar

    # TEST aggregates
    rel_terms_num, rel_terms_den = [], []
    nn_terms = []
    uv_test = []
    norms_test = []

    for (mu_i, nu_i, U_ref) in samples:
        U_ref = np.asarray(U_ref, dtype=np.float32)        # [T, N_h]
        U_pred = np.asarray(fw_pod_fn(
            torch.as_tensor(mu_i, dtype=torch.float32).unsqueeze(0) if np.ndim(mu_i) == 1 else mu_i,
            torch.as_tensor(nu_i, dtype=torch.float32).unsqueeze(0) if np.ndim(nu_i) == 1 else nu_i
        ), dtype=np.float32)                               # [T, N_h]

        Tm = min(U_ref.shape[0], U_pred.shape[0])
        U_ref = U_ref[:Tm]; U_pred = U_pred[:Tm]

        # norms & residuals per time
        ref_sq = g_norm_2(U_ref, G)                # [T]
        res_sq = g_norm_2(U_ref - U_pred, G)       # [T]

        # POD projection of TEST and "NN-only" deviation
        U_proj = g_project(A_P, G, U_ref)          # [T, N_h]
        nn_sq  = g_norm_2(U_pred - U_proj, G)      # [T]
        uv_sq  = g_norm_2(U_ref - U_proj, G)       # [T]

        rel_terms_num.append(float(res_sq.sum()))
        rel_terms_den.append(float(ref_sq.sum()))
        nn_terms.append(float(np.mean(nn_sq)))
        uv_test.append(float(np.mean(uv_sq)))

        # snapshot-wise collection for m/M (time sensitive)
        norms_test.extend([float(np.sqrt(v)) for v in ref_sq])

    total_num = float(np.sum(rel_terms_num))
    total_den = float(np.sum(rel_terms_den))
    E_R = float(np.sqrt(total_num) / (np.sqrt(total_den) + 1e-24))

    m = float(np.quantile(norms_test, 0.1))
    M = float(np.max(norms_test))

    uv_test_mean = float(np.mean(uv_test))
    E_POD      = float((unexplained_var_train ** 0.5) / (m + 1e-24))
    E_POD_inf  = float((uv_test_mean ** 0.5) / (m + 1e-24)) 
    E_S        = float((abs(uv_test_mean - unexplained_var_train) ** 0.5) / (m + 1e-24))
    E_NN       = float(np.sqrt(np.mean(nn_terms) / (np.mean(rel_terms_den) / len(samples) + 1e-24)))
    upper_bnd  = float(E_S + E_POD + E_NN)
    lower_bnd  = float((m / (M + 1e-24)) * E_POD_inf)

    return {
        "m": m, "M": M,
        "E_R": E_R,
        "E_S": E_S,
        "E_POD": E_POD,
        "E_POD_inf": E_POD_inf,
        "E_NN": E_NN,
        "upper_bound": upper_bnd,
        "lower_bound": lower_bnd,
    }

def dod_dl_error_decomposition(
    dod_matrix,
    fw_dod_fn,                          
    samples,
    N_prime,
    example_name: str,                       
    G: np.ndarray,
    T_end: float = 1.0,
):
    """
      E_R, E_S, E_DOD, E_DOD_inf, E_NN and bounds, averaged over (mu, t).
    """
    base = training_data_path(example_name)
    # ambient POD matrix A: [N_h, N_A]
    A = np.load(base / f'N_A_ambient_{example_name}.npz')['ambient'].astype(np.float32)
    # training FOM set to compute "unexplained variance" on TRAIN
    train_npz = np.load(base / f'full_order_training_data_{example_name}.npz')
    S_train = train_npz['solution'].astype(np.float32)   # [Ns_train, T, N_h]
    Mu_train = train_npz['mu'].astype(np.float32)        # [Ns_train, dim_mu]

    # decide device for dod_matrix inputs
    device = (next(dod_matrix.parameters()).device
              if hasattr(dod_matrix, "parameters") else torch.device("cpu"))

    def mu_to_tensor(mu_np):
        mu_t = torch.as_tensor(mu_np, dtype=torch.float32, device=device)
        return mu_t.unsqueeze(0) if mu_t.ndim == 1 else mu_t

    def t_to_tensor(t_idx: int, Tm: int):
        # map index -> physical/normalized time
        t_val = 0.0 if Tm <= 1 else (t_idx / (Tm - 1)) * T_end
        return torch.tensor([t_val], dtype=torch.float32, device=device)

    # unexplained variance on TRAIN via time-resolved DOD projector 
    uv_train_vals = []
    for mu_np, U in zip(Mu_train, S_train):
        Tm = U.shape[0]
        mu_t = mu_to_tensor(mu_np)
        for t_idx in range(Tm):
            u = U[t_idx]                               # [N_h]
            t_tensor = t_to_tensor(t_idx, Tm)          # [1]
            tilde_V = dod_matrix(mu_t, t_tensor)       # [N_A, N']
            tilde_V = _ensure_np(tilde_V).astype(np.float32)
            V = A @ tilde_V                            # [N_h, N']
            u_proj = g_project(V, G, u)
            uv_train_vals.append(g_norm_2(u - u_proj, G))
    unexplained_var_train = float(np.mean(uv_train_vals))

    # test accumulators 
    num_R, den_R = [], []
    nn_ratios = []     
    uv_test_vals = []
    norms_test = []

    for (mu_i, nu_i, U_ref) in samples:
        U_ref = _ensure_np(U_ref).astype(np.float32)    # [T, N_h]
        U_pred = _ensure_np(fw_dod_fn(
            torch.as_tensor(mu_i, dtype=torch.float32, device=device).unsqueeze(0)
                if np.ndim(mu_i) == 1 else
            torch.as_tensor(mu_i, dtype=torch.float32, device=device),
            torch.as_tensor(nu_i, dtype=torch.float32, device=device).unsqueeze(0)
                if np.ndim(nu_i) == 1 else
            torch.as_tensor(nu_i, dtype=torch.float32, device=device)
        )).astype(np.float32)                            # [T, N_h]

        Tm = min(U_ref.shape[0], U_pred.shape[0])
        mu_t = mu_to_tensor(mu_i)

        for t_idx in range(Tm):
            u = U_ref[t_idx]; y = U_pred[t_idx]
            t_tensor = t_to_tensor(t_idx, Tm)
            tilde_V = dod_matrix(mu_t, t_tensor)        # [N_A, N']
            tilde_V = _ensure_np(tilde_V).astype(np.float32)
            V = A @ tilde_V                              # [N_h, N']

            u_proj = g_project(V, G, u)

            ref_sq = g_norm_2(u, G)
            res_sq = g_norm_2(y - u, G)
            nn_sq  = g_norm_2(y - u_proj, G)
            uv_sq  = g_norm_2(u - u_proj, G)

            num_R.append(res_sq)
            den_R.append(ref_sq)
            nn_ratios.append(nn_sq)
            uv_test_vals.append(uv_sq)
            norms_test.append(np.sqrt(ref_sq))

    # aggregate over (mu, t)
    E_R = float(np.sqrt(np.sum(num_R)) / (np.sqrt(np.sum(den_R)) + 1e-24))
    m = float(np.quantile(norms_test, 0.1))
    M   = float(np.max(norms_test))

    # load in tail sum of singular values for fixed (\mu, t)
    path = training_data_path(example_name) / f"pod_singular_values_{example_name}.npz"
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        sigma_sup = np.asarray(blob['sigma_mu_t_sup'], dtype=np.float64).reshape(-1)
        N_A_file = int(sigma_sup.shape[0])
        if N_prime >= N_A_file:
            tail_sum = 0.0
        else:
            tail_sum = float(np.sum(sigma_sup[N_prime:]))
    else:
        raise FileExistsError('Singular values not found.')

    uv_test_mean = float(np.mean(uv_test_vals))
    E_DOD_inf  = float(np.sqrt(tail_sum) / (m + 1e-24))
    E_DOD      = float(np.sqrt(unexplained_var_train) / (m + 1e-24))
    E_S        = float(np.sqrt(abs(uv_test_mean - unexplained_var_train)) / (m + 1e-24))
    E_NN       = float(np.sqrt(np.mean(nn_ratios)) / (m + 1e-24))
    upper_bnd  = float(E_S + E_DOD + E_NN)
    lower_bnd  = float((m / (M + 1e-24)) * E_DOD_inf)

    return {
        "m": m, "M": M,
        "E_R": E_R,
        "E_S": E_S,
        "E_DOD": E_DOD,
        "E_DOD_inf": E_DOD_inf,
        "E_NN": E_NN,
        "upper_bound": upper_bnd,
        "lower_bound": lower_bnd,
    }

