'''
Data Gathering Script for example 04.
The usage of N and Nt can break the learning,
when not considered!!
====================================
IMPORTANT: The usage of specified N and Nt can break the learning,
           when not considered!!
           (Ns doesnt break it, but is important for notation)
====================================


Run Programm 
------------------------------
python3 examples/ex04/data.py 
[OPTIONALS]
--N "reduction number for POD"
--Ns "number of TOTAL parameter samples"
--Nt "number of time steps per solution"
-----------------------------
'''

import argparse
from pymor.basic import *
import numpy as np
from core.configs.parameters import Ex04Parameters  
from utils.paths import training_data_path

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
        training_data_path(example_name) / f"normalization_{tag}_{example_name}.npz",
        mu_min=mu_min, mu_max=mu_max, nu_min=nu_min, nu_max=nu_max,
        sol_min=sol_min, sol_max=sol_max
    )
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--Ns', type=int, default=None)
    parser.add_argument('--Nt', type=int, default=None)
    args = parser.parse_args()

    example_name = 'ex04'
    P = Ex04Parameters()

    if args.N is not None: P.N = args.N
    if args.Ns is not None: P.Ns = args.Ns
    if args.Nt is not None: P.Nt = args.Nt

    '''
    --------------------------
    Set up Problem
    --------------------------
    '''

    from core.bindings.fom import discretize

    default_config = {
        "reduction": 1e-20,
        "grid.dim": 2,
        "grid.yasp_x": P.grid_size,
        "grid.yasp_y": P.grid_size,
        "time.time": 0.0,
        "time.dt": P.dt,
        "time.solverSteps": 0.01,
        "time.T": P.T,
        "problem.eta": 0.2,
        "problem.inflowVelocity" : 1.0,
        "problem.non-parametric.openingHeight": 0.3,
        "problem.parametric.coatingHeight": 0.0,
        "problem.parametric.minPermeability": 0.0,
        "problem.parametric.coatingPermeability": 0.0,
        "problem.parametric.inflowAngle": 0.0,
        "darcy.reduction": 1e-12,
        "visualization.subsampling": 8,
        "visualization.subsamplingVelocity": 5,
        "visualization.subsamplingDG": 5,
    }

    fom = discretize(default_config)

    # Define the parameter space with ranges for 'mu' and 'nu'
    parameter_space = fom.parameters.space({
        'mu': (0., 1.),
        'nu': (0.4, 0.6)
    })

    # Generate training and validation sets
    sample_size_per_param = int(np.ceil(np.power(P.Ns, 1 / 4)))
    training_set = parameter_space.sample_uniformly(sample_size_per_param)

    # Collect parameter arrays
    mu_arr = np.array([p['mu'] for p in training_set], dtype=np.float32)
    nu_arr = np.array([p['nu'] for p in training_set], dtype=np.float32)

    # Enforce Dirichlet shift
    u_t_0 = fom.solve(parameter_space.sample_uniformly(1)[0])
    u_t_0_np = u_t_0.to_numpy().astype(np.float32)

    tdir = training_data_path(example_name)

    np.savez_compressed(tdir / f"dirichlet_shift_{example_name}.npz", 
                        ut0 = u_t_0_np)


    '''
    --------------------------
    Instationary Data
    --------------------------
    '''

    solutions = []
    shifted_solutions = []
    shifted_solutions_pymor = fom.solution_space.empty()

    for mu_nu in training_set:
        solution = fom.solve(mu_nu)  # [Nt, Nh]
        shifted_solution = solution - u_t_0
        shifted_solutions_pymor.append(shifted_solution)
        shifted_solutions.append(shifted_solution.to_numpy().astype(np.float32))
        solutions.append(solution.to_numpy().astype(np.float32))

    shifted_solutions = np.stack(shifted_solutions, axis=0)  # [Ns, Nt, Nh]
    solutions = np.stack(solutions, axis=0)                  # [Ns, Nt, Nh]

    # Gram matrix (compact)
    G = fom.l2_product.matrix.toarray().astype(np.float32)  # [Nh, Nh]
    np.savez_compressed(tdir / f'gram_matrix_{example_name}.npz', gram=G)

    # Save compact FOM data
    np.savez_compressed(tdir / f'full_order_training_data_{example_name}.npz',
                        mu=mu_arr, nu=nu_arr, solution=solutions)

    # --- Ambient for DOD-based ROMs ---
    pod_modes, singular_values = pod(shifted_solutions_pymor, product=fom.l2_product, modes=P.N_A)
    A = pod_modes.to_numpy().T.astype(np.float32)            # [Nh, N_A]
    ensure_modes(A, G, P.N_A)
    np.savez_compressed(tdir / f'N_A_ambient_{example_name}.npz', ambient=A)

    # --- Ambient for POD-based ROMs ---
    pod_modes_N, singular_values_N = pod(shifted_solutions_pymor, product=fom.l2_product, modes=P.N)
    A_P = pod_modes_N.to_numpy().T.astype(np.float32)  # [Nh, N]
    ensure_modes(A_P, G, P.N)
    np.savez_compressed(tdir / f'N_ambient_{example_name}.npz', ambient=A_P)

    # --- Sanity Check for G-orthonormality ---
    I_test = A.T @ G @ A
    print('||ATGA - I||_F =', np.linalg.norm(I_test - np.eye(I_test.shape[0])))
    I_test = A_P.T @ G @ A_P
    print('||ATGA - I||_F =', np.linalg.norm(I_test - np.eye(I_test.shape[0])))

    # --- Reduced instationary data: [Ns, Nt, N_A] ---
    GA = (G @ A).astype(np.float32)                               # [Nh, N_A]
    reduced = np.einsum('ijk,kl->ijl', shifted_solutions, GA)     # [Ns, Nt, N_A]
    np.savez_compressed(tdir / f'N_A_reduced_training_data_{example_name}.npz',
                        mu=mu_arr, nu=nu_arr, solution=reduced)

    # --- Reduced instationary data: [Ns, Nt, N] ---
    GA_P = (G @ A_P).astype(np.float32)                           # [Nh, N]
    reduced_P = np.einsum('ijk,kl->ijl', shifted_solutions, GA_P) # [Ns, Nt, N]
    np.savez_compressed(tdir / f'N_reduced_training_data_{example_name}.npz',
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
    N_nu_per_mu_t  = 30

    rng = np.random.default_rng(1234)

    mu_indices = rng.choice(Ns, size=min(N_mu_samples, Ns), replace=False)
    mu_candidates = mu_arr_2d[mu_indices]  # [N_mu_samples, p]

    all_t = np.arange(P.Nt, dtype=int)
    t_choices = [rng.choice(all_t, size=min(N_t_samples, len(all_t)), replace=False)
                for _ in range(len(mu_candidates))]

    nu_min = nu_arr_2d.min(axis=0)  # [q]
    nu_max = nu_arr_2d.max(axis=0)  # [q]

    sigma_mu_t_sup = np.zeros((P.N_A,), dtype=np.float32)

    for i, mu_vec in enumerate(mu_candidates):
        nu_batch = rng.uniform(nu_min, nu_max, size=(N_nu_per_mu_t, q)).astype(np.float32)
        nu_batch = nu_batch if nu_batch.ndim == 2 else nu_batch[:, None]

        nu_trajectories = []
        for k in range(N_nu_per_mu_t):
            params = fom.parameters.parse({
                'mu': mu_vec.astype(float).tolist(),
                'nu': nu_batch[k].astype(float).tolist() if q > 1 else float(nu_batch[k, 0])
            })
            traj = fom.solve(params)
            nu_trajectories.append(traj)

        for t_idx in t_choices[i]:
            snapshots = fom.solution_space.empty()
            for traj in nu_trajectories:
                snapshots.append(traj[t_idx])

            _, sv = pod(snapshots, product=fom.l2_product, modes=P.N_A)
            sv = np.asarray(sv, dtype=np.float32)
            if sv.shape[0] < P.N_A:
                sv = np.pad(sv, (0, P.N_A - sv.shape[0]))
            np.maximum(sigma_mu_t_sup, sv, out=sigma_mu_t_sup)

    sigma_global = np.asarray(singular_values, dtype=np.float32)
    if sigma_global.shape[0] < P.N_A:
        sigma_global = np.pad(sigma_global, (0, P.N_A - sigma_global.shape[0]))
    np.savez_compressed(tdir / f'pod_singular_values_{example_name}.npz',
                        sigma_global_NA=sigma_global,
                        sigma_mu_t_sup=sigma_mu_t_sup.astype(np.float32))

    '''
    --------------------------
    Save normalization meta data
    --------------------------
    '''

    _save_norm(example_name, "N_A_reduced", mu_arr, nu_arr, reduced)
    _save_norm(example_name, "N_reduced",   mu_arr, nu_arr, reduced_P)


if __name__ == '__main__':
    main()
