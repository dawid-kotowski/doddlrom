from pymor.basic import *
import numpy as np
from core.configs.parameters import Ex03Parameters  

P = Ex03Parameters()
example_name = 'ex03'

'''
====================================
Note: This example employs 
      N_h = N_A
====================================
'''

'''
------------------------
Helper Method
------------------------
'''

def ensure_modes(A, G, target: int):
    """
    Sanity check for POD
    """
    if A.shape[1] != target:
        print("Actual POD size: ", A.shape[1])
        raise AssertionError("Reduction too small")
    I_test = A.T @ G @ A
    err = np.linalg.norm(I_test - np.eye(I_test.shape[0]), ord='fro')
    if not np.isclose(err, 0, atol=1e-3):
        print("A.T G A - I = ", err)
        raise AssertionError("Not sufficiently orthonormal")
    
def build_pseudo_ambient(G: np.ndarray, jitter: float = 1e-12,
                                     dtype=np.float32) -> np.ndarray:
    """
    Return A in R^{N_h x N_h} with A^T G A = I_{N_h}.
    Uses Cholesky: G = L L^T  =>  A = L^{-T}.
    """
    Nh = G.shape[0]
    I = np.eye(Nh, dtype=G.dtype)

    # Robust Cholesky with tiny jitter if needed
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(G + jitter * I)

    # A = (L^T)^{-1}
    A = np.linalg.inv(L.T)

    # Sanity check (float64 first, then cast)
    Itest = A.T @ G @ A
    err = np.linalg.norm(Itest - I, ord='fro')
    if not np.isclose(err, 0.0, atol=1e-5):
        raise AssertionError(f"A^T G A deviates from identity: ||·||_F = {err:.3e}")

    return A.astype(dtype)

    

'''
--------------------------
Set up Problem
--------------------------
'''

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

# Discretize the stationary problem
stat_fom, stat_fom_data = discretize_stationary_cg(stationary_problem, diameter=P.diameter)

# Define the parameter space with ranges for 'mu' and 'nu'
parameter_space = fom.parameters.space({
    'mu': (0.2, 0.8),
    'nu': (1e-4, 1e-2)
})

# --- stationary FOM & shift ---
stat_parameter_space = stat_fom.parameters.space({
    'mu': (0.2, 0.8),
    'nu': (1e-4, 1e-2)
})

# Generate training and validation sets
sample_size_per_param = int(np.ceil(np.power(P.Ns, 1 / 2)))
training_set = parameter_space.sample_uniformly(sample_size_per_param)

# Collect parameter arrays
mu_arr = np.array([p['mu'] for p in training_set], dtype=np.float32)
nu_arr = np.array([p['nu'] for p in training_set], dtype=np.float32)

# Solve all stationary samples; make both a shifted solution_set for POD and a raw array for saving
stat_mu = np.array([p['mu'] for p in training_set], dtype=np.float32)
stat_nu = np.array([p['nu'] for p in training_set], dtype=np.float32)

# Create an empty list to hold the training data
solution_set = fom.solution_space.empty()         
stat_solution_set = stat_fom.solution_space.empty()

# Enforce Dirichlet shift
u_t_0 = fom.solve(parameter_space.sample_uniformly(1)[0])
u_t_0_np = u_t_0.to_numpy().astype(np.float32)
u_0 = stat_fom.solve(stat_parameter_space.sample_uniformly(1)[0])
u_0_np = u_0.to_numpy().astype(np.float32)
np.savez_compressed(f"examples/{example_name}/training_data/dirichlet_shift_{example_name}.npz", 
                    u0=u_0_np, ut0 = u_t_0_np)

'''
--------------------------
Instationary Data
--------------------------
'''

solutions = []
shifted_solutions = []
shifted_solutions_pymor = fom.solution_space.empty()
for mu_nu in training_set:
    solution = fom.solve(mu_nu)                                                     # [Nt, Nh]
    shifted_solution = solution - u_t_0   
    shifted_solutions_pymor.append(shifted_solution)                                  
    shifted_solutions.append(shifted_solution.to_numpy().astype(np.float32))
    solutions.append(solution.to_numpy().astype(np.float32))
shifted_solutions = np.stack(shifted_solutions, axis=0)
solutions = np.stack(solutions, axis=0)                                             # [Ns, Nt, Nh]
fom.visualize(solution)                                                             # visualize last solution as check

# Gram matrix (compact)
G = fom.h1_0_semi_product.matrix.toarray().astype(np.float32)  # [Nh, Nh]
np.savez_compressed(f'examples/{example_name}/training_data/gram_matrix_{example_name}.npz', gram=G)

# Save compact FOM data
np.savez_compressed(f'examples/{example_name}/training_data/full_order_training_data_{example_name}.npz',
                    mu=mu_arr, nu=nu_arr, solution=solutions)

# --- Ambient SKIPPED N_h for DOD-based ROMs ---
A = build_pseudo_ambient(G)
_, singular_values = pod(shifted_solutions_pymor, product=fom.h1_0_semi_product, modes=P.N_A)
ensure_modes(A, G, P.N_A)
np.savez_compressed(f'examples/{example_name}/training_data/N_A_ambient_{example_name}.npz', ambient=A)

# --- Ambient for POD-based ROMs ---
pod_modes_N, singular_values_N = pod(shifted_solutions_pymor, product=fom.h1_0_semi_product, modes=P.N)
A_P = pod_modes_N.to_numpy().T.astype(np.float32)            # [Nh, N]
ensure_modes(A_P, G, P.N)
np.savez_compressed(f'examples/{example_name}/training_data/N_ambient_{example_name}.npz', ambient=A_P)

# --- Sanity Check for G-orthonormality ---
I_test = A.T @ G @ A
print('||ATGA - I||_F =', np.linalg.norm(I_test - np.eye(I_test.shape[0])))
# --- Sanity Check for G-orthonormality ---
I_test = A_P.T @ G @ A_P
print('||ATGA - I||_F =', np.linalg.norm(I_test - np.eye(I_test.shape[0])))

# --- Reduced instationary data: [Ns, Nt, N_A] ---
# Compute GA once, then batch-matmul
GA = (G @ A).astype(np.float32)                           # [Nh, N_A]
reduced = np.einsum('ijk,kl->ijl', shifted_solutions, GA)         # [Ns, Nt, N_A]

np.savez_compressed(f'examples/{example_name}/training_data/N_A_reduced_training_data_{example_name}.npz',
                    mu=mu_arr, nu=nu_arr, solution=reduced)

# --- Reduced instationary data: [Ns, Nt, N] ---
# Compute GA_P once, then batch-matmul
GA_P = (G @ A_P).astype(np.float32)                           # [Nh, N]
reduced_P = np.einsum('ijk,kl->ijl', shifted_solutions, GA_P)         # [Ns, Nt, N]

np.savez_compressed(f'examples/{example_name}/training_data/N_reduced_training_data_{example_name}.npz',
                    mu=mu_arr, nu=nu_arr, solution=reduced_P)

'''
----------------------------
Stationary Data 
----------------------------
'''

stat_solutions = []
shifted_stat_solutions = []
shifted_stat_solutions_pymor = stat_fom.solution_space.empty()
for mu_nu in training_set:
    stat_solution = stat_fom.solve(mu_nu)                                                     # [Nh]
    shifted_stat_solution = stat_solution - u_0   
    shifted_stat_solutions_pymor.append(shifted_stat_solution)                                  
    shifted_stat_solutions.append(shifted_stat_solution.to_numpy().astype(np.float32))
    stat_solutions.append(stat_solution.to_numpy().astype(np.float32))
shifted_stat_solutions = np.stack(shifted_stat_solutions, axis=0)
stat_solutions = np.stack(stat_solutions, axis=0)                                             # [Ns, Nh]
fom.visualize(stat_solution)                                                             # visualize last solution as check

# Save compact stationary FOM data (raw, unshifted)
np.savez_compressed(f'examples/{example_name}/training_data/stationary_training_data_{example_name}.npz',
                    mu=stat_mu, nu=stat_nu, solution=stat_solutions)

stat_G = stat_fom.h1_0_semi_product.matrix.toarray().astype(np.float32)       # [Nh, Nh]
np.savez_compressed(f'examples/{example_name}/training_data/stationary_gram_matrix_{example_name}.npz',
                    gram=stat_G)

# --- stationary POD/ambient/gram ---
stat_A = build_pseudo_ambient(stat_G)
ensure_modes(stat_A, stat_G, P.N_A)
np.savez_compressed(f'examples/{example_name}/training_data/stationary_ambient_matrix_{example_name}.npz',
                    ambient=stat_A)

# --- reduced stationary data: [N, N_A] ---
GA_stat = (stat_G @ stat_A).astype(np.float32)                         # [Nh, N_A]
stat_reduced = shifted_stat_solutions @ GA_stat                        # [Ns, N_A]

np.savez_compressed(f'examples/{example_name}/training_data/reduced_stationary_training_data_{example_name}.npz',
                    mu=stat_mu, nu=stat_nu, solution=stat_reduced)

'''
 -------------------------
 Singular value exports 
 -------------------------
'''

Ns = mu_arr.shape[0]
mu_arr_2d = mu_arr if mu_arr.ndim == 2 else mu_arr[:, None]
nu_arr_2d = nu_arr if nu_arr.ndim == 2 else nu_arr[:, None]
p = mu_arr_2d.shape[1]
q = nu_arr_2d.shape[1]

# Tunables
N_mu_samples   = 2
N_t_samples    = 2
N_nu_per_mu_t  = 20

rng = np.random.default_rng(1234)

mu_indices = rng.choice(Ns, size=min(N_mu_samples, Ns), replace=False)
mu_candidates = mu_arr_2d[mu_indices]                          # [N_mu_samples, p]

all_t = np.arange(P.Nt, dtype=int)
t_choices = [rng.choice(all_t, size=min(N_t_samples, len(all_t)), replace=False)
             for _ in range(len(mu_candidates))]

nu_min = nu_arr_2d.min(axis=0)                                  # [q]
nu_max = nu_arr_2d.max(axis=0)                                  # [q]

sigma_mu_t_sup = np.zeros((P.N_A,), dtype=np.float32)

for i, mu_vec in enumerate(mu_candidates):
    nu_batch = rng.uniform(nu_min, nu_max, size=(N_nu_per_mu_t, q)).astype(np.float32)
    nu_batch = nu_batch if nu_batch.ndim == 2 else nu_batch[:, None]

    nu_trajectories = []
    for k in range(N_nu_per_mu_t):
        params = fom.parameters.parse({
            'mu': mu_vec.astype(float).tolist(),
            'nu': nu_batch[k].astype(float).tolist() if q > 1 else float(nu_batch[k,0])
        })
        traj = fom.solve(params)         
        nu_trajectories.append(traj)

    for t_idx in t_choices[i]:
        snapshots = fom.solution_space.empty()
        for traj in nu_trajectories:
            snapshots.append(traj[t_idx])

        _, sv = pod(snapshots, product=fom.h1_0_semi_product, modes=P.N_A)
        sv = np.asarray(sv, dtype=np.float32)
        if sv.shape[0] < P.N_A:
            sv = np.pad(sv, (0, P.N_A - sv.shape[0]))
        np.maximum(sigma_mu_t_sup, sv, out=sigma_mu_t_sup)

np.savez_compressed(f'examples/{example_name}/training_data/pod_singular_values_{example_name}.npz',
                    sigma_global_NA=singular_values.astype(np.float32),
                    sigma_mu_t_sup=sigma_mu_t_sup.astype(np.float32))
