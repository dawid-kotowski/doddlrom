from pymor.basic import *
import numpy as np
from master_project_1.configs.ex01_parameters import Ex01Parameters  

P = Ex01Parameters()
example_name = 'ex01'

# Define the advection function dependent on 'mu'
def advection_function(x, mu):
    mu_value = mu['mu']
    return np.array([[np.cos(mu_value)*30, np.sin(mu_value)*30] for _ in range(x.shape[0])])

# Define the stationary problem
advection_params = Parameters({'mu': 1})
advection_generic_function = GenericFunction(advection_function, dim_domain=2, 
                                             shape_range=(2,), parameters=advection_params)
stationary_problem = StationaryProblem(
    domain=RectDomain(),
    rhs=ExpressionFunction('0', 2),
    diffusion=LincombFunction(
        [ExpressionFunction('1 - x[0]', 2), ExpressionFunction('x[0]', 2)],
        [ProjectionParameterFunctional('nu', 1), 1]
    ),
    dirichlet_data=ExpressionFunction('(-(x[1] - 0.5)**2 + 0.25) * (x[0] < 1e-10)', 2),
    advection=advection_generic_function,
    name='advection_problem'
)

# Define the instationary problem
problem = InstationaryProblem(
    T=P.T,
    initial_data=ConstantFunction(0., 2),
    stationary_part=stationary_problem,
    name='advection_problem'
)

'''
--------------------------
Instationary Training Data for the POD-DL-ROM inspired NN
--------------------------
'''

# Discretize the problem
fom, fom_data = discretize_instationary_cg(problem, diameter=P.diameter, nt=P.Nt)

# Define the parameter space with ranges for 'mu' and 'nu'
parameter_space = fom.parameters.space({'nu': (0.5, 1), 'mu': (0.2, np.pi-0.2)})

# Create an empty list to hold the training data
training_data = []
solution_set = fom.solution_space.empty()

# Generate training and validation sets
sample_size_per_param = int(np.ceil(np.sqrt(P.Ns)))
training_set = parameter_space.sample_uniformly(sample_size_per_param)

# Collect parameter arrays
mu_arr = np.array([p['mu'] for p in training_set], dtype=np.float32)
nu_arr = np.array([p['nu'] for p in training_set], dtype=np.float32)

# Solve all samples once; stack to [Ns, Nt, Nh], float32
solutions = []
for mu_nu in training_set:
    solution = fom.solve(mu_nu)
    sol = solution.to_numpy()                   # shape [Nt, Nh]
    solution_set.append(solution)
    solutions.append(sol.astype(np.float32))
solutions = np.stack(solutions, axis=0)         # [Ns, Nt, Nh]
fom.visualize(solution)                         # visualize last solution as check

# Save compact FOM data
np.savez_compressed(f'examples/{example_name}/training_data/full_order_training_data_ex01.npz',
                    mu=mu_arr, nu=nu_arr, solution=solutions)

# --- Ambient for DOD-based ROMs ---
pod_modes, singular_values = pod(solution_set, product=fom.h1_0_semi_product, modes=P.N_A)
A = pod_modes.to_numpy().T.astype(np.float32)            # [Nh, N_A]
np.savez_compressed(f'examples/{example_name}/training_data/N_A_ambient_ex01.npz', ambient=A)

# --- Ambient for POD-based ROMs ---
pod_modes_N, singular_values_N = pod(solution_set, product=fom.h1_0_semi_product, modes=P.N)
A_P = pod_modes_N.to_numpy().T.astype(np.float32)            # [Nh, N]
np.savez_compressed(f'examples/{example_name}/training_data/N_ambient_ex01.npz', ambient=A_P)

# Gram matrix (compact)
G = fom.h1_0_semi_product.matrix.toarray().astype(np.float32)  # [Nh, Nh]
np.savez_compressed(f'examples/{example_name}/training_data/gram_matrix_ex01.npz', gram=G)

# --- Reduced instationary data: [Ns, Nt, N_A] ---
# Compute GA once, then batch-matmul
GA = (G @ A).astype(np.float32)                           # [Nh, N_A]
reduced = np.einsum('ijk,kl->ijl', solutions, GA)         # [Ns, Nt, N_A]

np.savez_compressed(f'examples/{example_name}/training_data/N_A_reduced_training_data_ex01.npz',
                    mu=mu_arr, nu=nu_arr, solution=reduced)

# --- Reduced instationary data: [Ns, Nt, N] ---
# Compute GA_P once, then batch-matmul
GA_P = (G @ A_P).astype(np.float32)                           # [Nh, N]
reduced_P = np.einsum('ijk,kl->ijl', solutions, GA_P)         # [Ns, Nt, N]

np.savez_compressed(f'examples/{example_name}/training_data/N_reduced_training_data_ex01.npz',
                    mu=mu_arr, nu=nu_arr, solution=reduced_P)

'''
 -------------------------
 Singular value exports 
 -------------------------
'''

# TUNABLES to keep cost low
N_mu_samples   = 2    # number of distinct mu's to consider
N_t_samples    = 2    # number of time indices per mu (randomly chosen from 0..Nt-1)
N_nu_per_mu_t  = 20   # number of nu values per (mu, t)

rng = np.random.default_rng(1234)
mu_low, mu_high = 0.2, np.pi - 0.2
nu_low, nu_high = 0.5, 1.0

mu_candidates = rng.uniform(mu_low, mu_high, size=N_mu_samples).astype(np.float32)
all_t = np.arange(P.Nt, dtype=int)
t_choices = [rng.choice(all_t, size=N_t_samples, replace=False) for _ in range(N_mu_samples)]

# initialize the running supremum vector (length N_A)
sigma_mu_t_sup = np.zeros((P.N_A,), dtype=np.float32)

for i, mu_val in enumerate(mu_candidates):
    # for each chosen mu, pre-sample a small batch of nu values
    nu_batch = rng.uniform(nu_low, nu_high, size=N_nu_per_mu_t).astype(np.float32)

    nu_trajectories = []
    for nu_val in nu_batch:
        params = fom.parameters.parse({'mu': float(mu_val), 'nu': float(nu_val)})
        traj = fom.solve(params)  
        nu_trajectories.append(traj)

    Nt_from_solver = len(nu_trajectories[0])
    def solver_t_idx(t_idx: int) -> int:
        return t_idx  

    for t_idx in t_choices[i]:
        snapshots = fom.solution_space.empty()
        for traj in nu_trajectories:
            snapshots.append(traj[solver_t_idx(t_idx)])  # 1-snapshot append per nu

        # POD across nu at fixed (mu, t)
        _, sv = pod(snapshots, product=fom.h1_0_semi_product, modes=P.N_A) 
        sv = np.asarray(sv, dtype=np.float32)
        if sv.shape[0] < P.N_A:
            sv = np.pad(sv, (0, P.N_A - sv.shape[0]))  # zero pad to N_A

        np.maximum(sigma_mu_t_sup, sv, out=sigma_mu_t_sup)

# Save 
np.savez_compressed(f'examples/{example_name}/training_data/pod_singular_values_ex01.npz',
                    sigma_global_NA=singular_values.astype(np.float32),  # [N_A]
                    sigma_mu_t_sup=sigma_mu_t_sup.astype(np.float32))    # [N_A]

'''
----------------------------
Stationary Data for the CoLoRA inspired NN
----------------------------
'''

# Discretize the stationary problem
stat_fom, stat_fom_data = discretize_stationary_cg(stationary_problem, diameter=P.diameter)

# --- stationary FOM & shift ---
stat_parameter_space = stat_fom.parameters.space({'nu': (0.5, 1), 'mu': (0.2, np.pi-0.2)})

# Dirichlet shift (store compressed)
u_0 = stat_fom.solve(stat_parameter_space.sample_uniformly(1)[0])
u_0_np = u_0.to_numpy().astype(np.float32)
np.savez_compressed(f"examples/{example_name}/training_data/dirichlet_shift_ex01.npz", u0=u_0_np)

# Solve all stationary samples; make both a shifted solution_set for POD and a raw array for saving
stat_mu = np.array([p['mu'] for p in training_set], dtype=np.float32)
stat_nu = np.array([p['nu'] for p in training_set], dtype=np.float32)

stat_solutions = []          
stat_solution_set = stat_fom.solution_space.empty()

for mu_nu in training_set:
    sol_vec = stat_fom.solve(mu_nu)                         # vector in model space
    sol_np = sol_vec.to_numpy().flatten().astype(np.float32)  # [Nh]
    stat_solutions.append(sol_np)

    # shift for POD
    stat_solution_set.append(sol_vec - u_0)

stat_solutions = np.stack(stat_solutions, axis=0)              # [Ns, Nh]

# Save compact stationary FOM data (raw, unshifted)
np.savez_compressed(f'examples/{example_name}/training_data/stationary_training_data_ex01.npz',
                    mu=stat_mu, nu=stat_nu, solution=stat_solutions)

# --- stationary POD/ambient/gram ---
stat_pod_modes, stat_singular_values = pod(stat_solution_set,
                                           product=stat_fom.h1_0_semi_product, modes=P.N_A)
stat_A = stat_pod_modes.to_numpy().T.astype(np.float32)       # [Nh, N_A]
np.savez_compressed(f'examples/{example_name}/training_data/stationary_ambient_matrix_ex01.npz',
                    ambient=stat_A)

stat_G = stat_fom.h1_0_semi_product.matrix.toarray().astype(np.float32)  # [Nh, Nh]
np.savez_compressed(f'examples/{example_name}/training_data/stationary_gram_matrix_ex01.npz',
                    gram=stat_G)

# --- reduced stationary data: [N, N_A] ---
GA_stat = (stat_G @ stat_A).astype(np.float32)                 # [Nh, N_A]
stat_reduced = stat_solutions @ GA_stat                        # [Ns, N_A]

np.savez_compressed(f'examples/{example_name}/training_data/reduced_stationary_training_data_ex01.npz',
                    mu=stat_mu, nu=stat_nu, solution=stat_reduced)
