from pymor.basic import *
import numpy as np
from core.configs.parameters import Ex02Parameters  

P = Ex02Parameters()
example_name = 'ex02'

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
    name='advection_problem'
)

# Define the instationary problem
problem = InstationaryProblem(
    T=1.,
    initial_data=ConstantFunction(0, 2),
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

# Discretize the problem
fom, fom_data = discretize_instationary_cg(problem, diameter=P.diameter, nt=P.Nt)

print(fom.parameters)

# Define the parameter space with ranges for 'mu' and 'nu'
parameter_space = fom.parameters.space({
    'mu': (0.3, 0.7),
    'nu': (0.002, 0.005)
})

# Generate training and validation sets
sample_size_per_param = int(np.ceil(np.power(P.Ns, 1 / 4)))
training_set = parameter_space.sample_uniformly(sample_size_per_param)

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
np.savez_compressed(f'examples/{example_name}/training_data/full_order_training_data_{example_name}.npz',
                    mu=mu_arr, nu=nu_arr, solution=solutions)

# --- Ambient for DOD-based ROMs ---
pod_modes, singular_values = pod(solution_set, product=fom.h1_0_semi_product, modes=P.N_A)
A = pod_modes.to_numpy().T.astype(np.float32)            # [Nh, N_A]
np.savez_compressed(f'examples/{example_name}/training_data/N_A_ambient_{example_name}.npz', ambient=A)

# --- Ambient for POD-based ROMs ---
pod_modes_N, singular_values_N = pod(solution_set, product=fom.h1_0_semi_product, modes=P.N)
A_P = pod_modes_N.to_numpy().T.astype(np.float32)            # [Nh, N]
np.savez_compressed(f'examples/{example_name}/training_data/N_ambient_{example_name}.npz', ambient=A_P)

# Gram matrix (compact)
G = fom.h1_0_semi_product.matrix.toarray().astype(np.float32)  # [Nh, Nh]
np.savez_compressed(f'examples/{example_name}/training_data/gram_matrix_{example_name}.npz', gram=G)

# --- Reduced instationary data: [Ns, Nt, N_A] ---
# Compute GA once, then batch-matmul
GA = (G @ A).astype(np.float32)                           # [Nh, N_A]
reduced = np.einsum('ijk,kl->ijl', solutions, GA)         # [Ns, Nt, N_A]

np.savez_compressed(f'examples/{example_name}/training_data/N_A_reduced_training_data_{example_name}.npz',
                    mu=mu_arr, nu=nu_arr, solution=reduced)

# --- Reduced instationary data: [Ns, Nt, N] ---
# Compute GA_P once, then batch-matmul
GA_P = (G @ A_P).astype(np.float32)                           # [Nh, N]
reduced_P = np.einsum('ijk,kl->ijl', solutions, GA_P)         # [Ns, Nt, N]

np.savez_compressed(f'examples/{example_name}/training_data/N_reduced_training_data_{example_name}.npz',
                    mu=mu_arr, nu=nu_arr, solution=reduced_P)

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
----------------------------
Stationary Data for the CoLoRA inspired NN
----------------------------
'''

# Discretize the stationary problem
stat_fom, stat_fom_data = discretize_stationary_cg(stationary_problem, diameter=P.diameter)

# --- stationary FOM & shift ---
stat_parameter_space = stat_fom.parameters.space({
    'mu': (0.3, 0.7),
    'nu': (0.002, 0.005)
})

# Dirichlet shift (store compressed)
u_0 = stat_fom.solve(stat_parameter_space.sample_uniformly(1)[0])
u_0_np = u_0.to_numpy().astype(np.float32)
np.savez_compressed(f"examples/{example_name}/training_data/dirichlet_shift_{example_name}.npz", u0=u_0_np)

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
np.savez_compressed(f'examples/{example_name}/training_data/stationary_training_data_{example_name}.npz',
                    mu=stat_mu, nu=stat_nu, solution=stat_solutions)

# --- stationary POD/ambient/gram ---
stat_pod_modes, stat_singular_values = pod(stat_solution_set,
                                           product=stat_fom.h1_0_semi_product, modes=P.N_A)
stat_A = stat_pod_modes.to_numpy().T.astype(np.float32)       # [Nh, N_A]
np.savez_compressed(f'examples/{example_name}/training_data/stationary_ambient_matrix_{example_name}.npz',
                    ambient=stat_A)

stat_G = stat_fom.h1_0_semi_product.matrix.toarray().astype(np.float32)  # [Nh, Nh]
np.savez_compressed(f'examples/{example_name}/training_data/stationary_gram_matrix_{example_name}.npz',
                    gram=stat_G)

# --- reduced stationary data: [N, N_A] ---
GA_stat = (stat_G @ stat_A).astype(np.float32)                 # [Nh, N_A]
stat_reduced = stat_solutions @ GA_stat                        # [Ns, N_A]

np.savez_compressed(f'examples/{example_name}/training_data/reduced_stationary_training_data_{example_name}.npz',
                    mu=stat_mu, nu=stat_nu, solution=stat_reduced)
