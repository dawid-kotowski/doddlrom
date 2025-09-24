from pymor.basic import *
import numpy as np
from core.configs.parameters import Ex02Parameters  

P = Ex02Parameters()
example_name = 'ex02'

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
    

'''
--------------------------
Set up Problem
--------------------------
'''

# Define the advection function dependent on 'mu'
def advection_function(x, mu):
    mu_value = mu['mu'][2]
    return np.array([[np.cos(np.pi/(100*mu_value)), np.sin(np.pi/(100*mu_value))] for _ in range(x.shape[0])])
    
# Define rhs 
def rhs_function(x, mu):
    mu_values_1 = mu['mu'][0]
    mu_values_2 = mu['mu'][1]
    x0 = x[:, 0]
    x1 = x[:, 1]
    values = 10 * np.exp(-((x0 - mu_values_1)**2 + (x1 - mu_values_2)**2) / 0.07**2)
    return values

# Define the stationary problem
# Note, that the Dirichlet Boundary conditions are NOT enforced here, so the Dirichlet shift can be skipped!
mu_param = Parameters({'mu': 3, 'nu': 1})
advection_generic_function = GenericFunction(advection_function, dim_domain=2, shape_range=(2,), parameters=mu_param)
rhs_generic_function = GenericFunction(rhs_function, dim_domain=2, shape_range=(), parameters=mu_param)
stationary_problem = StationaryProblem(
    domain=RectDomain(),
    rhs=rhs_generic_function,
    diffusion=LincombFunction([ExpressionFunction('1', 2)],
                              [ProjectionParameterFunctional('nu', 1)]),
    advection=advection_generic_function,
    neumann_data=ConstantFunction(0, 2),
    dirichlet_data=None,
    name='nonlinear_wind_ex02'
)

# Define the instationary problem
problem = InstationaryProblem(
    T=1.,
    initial_data=ConstantFunction(0, 2),
    stationary_part=stationary_problem,
    name='nonlinear_wind_ex02'
)

# Discretize the problem
fom, fom_data = discretize_instationary_cg(problem, diameter=P.diameter, nt=P.Nt)

# Discretize the stationary problem
stat_fom, stat_fom_data = discretize_stationary_cg(stationary_problem, diameter=P.diameter)

# Define the parameter space with ranges for 'mu' and 'nu'
parameter_space = fom.parameters.space({
    'mu': (0.3, 0.7),
    'nu': (0.002, 0.005)
})

# --- stationary FOM & shift ---
stat_parameter_space = stat_fom.parameters.space({
    'mu': (0.3, 0.7),
    'nu': (0.002, 0.005)
})

# Generate training and validation sets
sample_size_per_param = int(np.ceil(np.power(P.Ns, 1 / 4)))
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

# --- Ambient for DOD-based ROMs ---
pod_modes, singular_values = pod(shifted_solutions_pymor, product=fom.h1_0_semi_product, l2_err=1e-5,
                                 modes=P.N_A)
A = pod_modes.to_numpy().T.astype(np.float32)            # [Nh, N_A]
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
stat_pod_modes, stat_singular_values = pod(shifted_stat_solutions_pymor,
                                           product=stat_fom.h1_0_semi_product, modes=P.N_A)
stat_A = stat_pod_modes.to_numpy().T.astype(np.float32)                # [Nh, N_A]
#ensure_modes(stat_A, stat_G, P.N_A)
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


'''
--------------------------
Save normalization meta data
--------------------------
'''

def _as2d(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 2 else x[:, None]

def _save_norm(example_name: str, tag: str, mu: np.ndarray, nu: np.ndarray, sol: np.ndarray):
    mu2 = _as2d(mu).astype(np.float32)  # [Ns, p]
    nu2 = _as2d(nu).astype(np.float32)  # [Ns, q]

    mu_min = mu2.min(axis=0).astype(np.float32); mu_max = mu2.max(axis=0).astype(np.float32)
    nu_min = nu2.min(axis=0).astype(np.float32); nu_max = nu2.max(axis=0).astype(np.float32)

    sol_flat = sol.reshape(-1, sol.shape[-1])    # [(Ns*Nt), D]
    sol_min = sol_flat.min(axis=0).astype(np.float32)
    sol_max = sol_flat.max(axis=0).astype(np.float32)
    np.savez_compressed(
        f"examples/{example_name}/training_data/normalization_{tag}_{example_name}.npz",
        mu_min=mu_min, mu_max=mu_max, nu_min=nu_min, nu_max=nu_max,
        sol_min=sol_min, sol_max=sol_max
    )

_save_norm(example_name, "N_A_reduced", mu_arr, nu_arr, reduced)

_save_norm(example_name, "N_reduced",   mu_arr, nu_arr, reduced_P)

_save_norm(example_name, "reduced_stationary", stat_mu, stat_nu, stat_reduced)