from pymor.basic import *
from core import reduced_order_models as rom
from core.configs.parameters import Ex03Parameters  
import numpy as np
import torch

#region --- Configure this run ------------------------------------------------------
example_name = 'ex03'
P = Ex03Parameters(profile="baseline")          # or "wide"/"tiny"/"debug"
P.assert_consistent()

#region --- Set up of FOM for pymor utility------------------------------------------
data = np.load(f'examples/{example_name}/training_data/full_order_training_data_{example_name}.npz')
mu = data['mu']          # shape [Ns, p]
nu = data['nu']          # shape [Ns, q]
solution = data['solution']  # shape [Ns, P.Nt, Nh]
Ns = mu.shape[0]
idx = np.arange(Ns)
np.random.shuffle(idx)
sel = idx[:3]
training_data = [(mu[i], nu[i], solution[i]) for i in sel]  


# Domain and parameters
domain = LineDomain([0., 1.], left='dirichlet', right='dirichlet')  # 1D interval

# Diffusion coefficient
diffusion_fn = LincombFunction(
    [ExpressionFunction('1.', 1)],           
    [ProjectionParameterFunctional('nu')]    
)

# Zero RHS and Dirichlet data
rhs = ConstantFunction(0.0, 1)
gD  = ConstantFunction(0.0, 1)

# Initial condition: smoothed step centered at mu
# u0(x; mu) = 0.5*(1 + tanh((x - mu)/eps))
def ic_step(x, mu):
    x0  = float(mu['mu'])
    eps = 0.01   
    v = 0.5 * (1.0 + np.tanh((x[:, 0] - x0) / eps))
    return v

initial_data = GenericFunction(ic_step, dim_domain=1, shape_range=(),
                               parameters=Parameters({'mu': 1}))

# Stationary part
stationary_problem = StationaryProblem(
    domain=domain,
    diffusion=diffusion_fn,
    rhs=rhs,
    dirichlet_data=gD,
    name='heat_ex03'
)

# Instationary problem
problem = InstationaryProblem(
    T=P.T,
    initial_data=initial_data,
    stationary_part=stationary_problem,
    name='heat_ex03'
)

# Discretize the problem
fom, fom_data = discretize_instationary_cg(problem, diameter=P.diameter, nt=P.Nt)


#endregion

# --- Set up solutions --------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
true_solution = fom.solution_space.empty()
G = np.load(f'examples/{example_name}/training_data/gram_matrix_{example_name}.npz')['gram'].astype(np.float32)

for entry in training_data:
    mu, nu, sol = entry

    def to_batch_vec(x):
        x_np = np.asarray(x)
        if x_np.ndim == 0:
            x_np = x_np[None]         
        return torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)

    mu_i = to_batch_vec(mu)
    nu_i = to_batch_vec(nu)

    # True solution (VectorArray)
    u_i = fom.solution_space.from_numpy(sol)
    true_solution.append(u_i)

    #region Check for normalization procedure
    A_NA = np.load(f'examples/{example_name}/training_data/N_A_ambient_{example_name}.npz')['ambient'].astype(np.float32)
    stats = np.load(f'examples/{example_name}/training_data/normalization_N_A_reduced_{example_name}.npz')
    sol_min = stats['sol_min'].astype(np.float32)  # [N_A]
    sol_max = stats['sol_max'].astype(np.float32)  # [N_A]
    U = sol.astype(np.float32)                              # [Nt, N_h]

    # Project: Y = A^T G u_t  -> [Nt, N_A]
    Y = np.einsum('ih,hj,tj->ti', A_NA.T, G, U, optimize=True)
    
    # Normalize
    eps = 1e-8
    Y_norm = (Y - sol_min[None, :]) / (sol_max[None, :] - sol_min[None, :] + eps)  # [Nt, N_A]

    # Denormalize
    Y_den = rom.denormalize_solution(
        torch.tensor(Y_norm, dtype=torch.float32, device=device),
        example_name, reduction_tag='N_A_reduced'
    ).cpu().numpy()  # [Nt, N_A]

    # Lift: û_t = A Y_den
    U_hat  = np.einsum('hi,ti->th', A_NA, Y_den, optimize=True)

    def _g_sq(X, Gmat):
        v = np.einsum('ti,ij,tj->t', X, Gmat, X, optimize=False)
        return np.maximum(v, 0.0)

    err_sq = _g_sq(U - U_hat, G)
    ref_sq = _g_sq(U, G)
    abs_err = float(np.sqrt(err_sq.mean()))
    rel_err = float(np.sqrt(err_sq.sum()) / (np.sqrt(ref_sq.sum()) + 1e-24))
    print(f"[ex03 | A^T G → norm → denorm → A]  abs={abs_err:.3e}  rel={rel_err:.3e}")
    #endregion! Check

    # inner DOD used by DOD+DFNN and DOD-DL-ROM
    innerDOD_model = rom.innerDOD(**P.make_innerDOD_kwargs()).to(device)

    # DOD+DFNN (DFNN -> N')
    dfnn_nprime = rom.DFNN(**P.make_dod_dfnn_DFNN_kwargs()).to(device)

    # DOD-DL-ROM (DFNN -> n, AE: N'<->n)
    coeff_n = rom.DFNN(**P.make_dod_dl_DFNN_kwargs()).to(device)
    enc = rom.Encoder(**P.make_dod_dl_Encoder_kwargs()).to(device)
    dec = rom.Decoder(**P.make_dod_dl_Decoder_kwargs()).to(device)

    # POD-DL-ROM (DFNN -> n, AE: N_A<->n)
    pod_coeff = rom.DFNN(**P.make_pod_DFNN_kwargs()).to(device)
    pod_enc = rom.Encoder(**P.make_pod_Encoder_kwargs()).to(device)
    pod_dec = rom.Decoder(**P.make_pod_Decoder_kwargs()).to(device)

    models = {
        "DOD+DFNN": {"inner": innerDOD_model, "coeff": dfnn_nprime},
        "DOD-DL-ROM": {"inner": innerDOD_model, "coeff": coeff_n, "enc": enc, "dec": dec},
        "POD-DL-ROM": {"coeff": pod_coeff, "enc": pod_enc, "dec": pod_dec}
    }

    fw = rom.forward_wrappers(P, device, models, example_name)

    _, pod_dl_residual, pod_dl_sol = rom.evaluate_rom_forward(
                        'POD-DL-ROM', fw['POD-DL-ROM'], (mu_i, nu_i), sol, G
                    )
    pod_dl_sol_vec = fom.solution_space.from_numpy(pod_dl_sol)
    _, dod_dl_residual, dod_dl_sol = rom.evaluate_rom_forward(
                        'DOD-DL-ROM', fw['DOD-DL-ROM'], (mu_i, nu_i), sol, G
                    )
    dod_dl_sol_vec = fom.solution_space.from_numpy(dod_dl_sol)
    _, dod_dfnn_residual, dod_dfnn_sol = rom.evaluate_rom_forward(
                        'DOD+DFNN', fw['DOD+DFNN'], (mu_i, nu_i), sol, G
                    )
    dod_dfnn_sol_vec = fom.solution_space.from_numpy(dod_dfnn_sol)


    # Visualize
    fom.visualize((u_i, dod_dfnn_sol_vec, u_i - dod_dfnn_sol_vec),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()}, ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'DOD+DFNN', f"Relative L\u00b2 error: mean={dod_dfnn_residual:.3e}"))
    fom.visualize((u_i, pod_dl_sol_vec, u_i - pod_dl_sol_vec),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'POD-DL-ROM', f"Relative L\u00b2 error: mean={pod_dl_residual:.3e}"))
    fom.visualize((u_i, dod_dl_sol_vec, u_i - dod_dl_sol_vec),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'DOD-DL-ROM', f"Relative L\u00b2 error: mean={dod_dl_residual:.3e}"))
