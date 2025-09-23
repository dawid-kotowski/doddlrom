from pymor.basic import *
from core import reduced_order_models as rom
from core.configs.parameters import Ex02Parameters  
import numpy as np
import torch

#region --- Configure this run ------------------------------------------------------
example_name = 'ex02'
P = Ex02Parameters(profile="baseline")          # or "wide"/"tiny"/"debug"
P.assert_consistent()

#region --- Set up of FOM for pymor utility------------------------------------------
data = np.load(f'examples/{example_name}/training_data/full_order_training_data_{example_name}.npz')
mu = data['mu']          # shape [Ns, p]
nu = data['nu']          # shape [Ns, q]
solution = data['solution']  # shape [Ns, P.Nt, Nh]
Ns = mu.shape[0]
idx = np.arange(Ns)
np.random.shuffle(idx)
sel = idx[:1]
training_data = [(mu[i], nu[i], solution[i]) for i in sel]  


# Define Full Order Model again
def advection_function(x, mu):
    mu_value = mu['mu'][2]
    return np.array([[np.cos(np.pi/(100*mu_value)), np.sin(np.pi/(100*mu_value))] for _ in range(x.shape[0])])

def rhs_function(x, mu):
    mu_values_1 = mu['mu'][0]
    mu_values_2 = mu['mu'][1]
    x0 = x[:, 0]
    x1 = x[:, 1]
    values = 10 * np.exp(-((x0 - mu_values_1)**2 + (x1 - mu_values_2)**2) / 0.07**2)
    return values

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
problem = InstationaryProblem(
    T=1.,
    initial_data=ConstantFunction(0, 2),
    stationary_part=stationary_problem,
    name='advection_problem'
)
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

    # CoLoRA (statDOD * statHadamard -> n ->(time-sensitive) N_A)
    stat_dod = rom.statDOD(**P.make_statDOD_kwargs())
    stat_coeff = rom.statHadamardNN(**P.make_statHadamard_kwargs())
    colora_coeff = rom.CoLoRA(**P.make_CoLoRA_kwargs())

    models = {
        "DOD+DFNN": {"inner": innerDOD_model, "coeff": dfnn_nprime},
        "DOD-DL-ROM": {"inner": innerDOD_model, "coeff": coeff_n, "enc": enc, "dec": dec},
        "POD-DL-ROM": {"coeff": pod_coeff, "enc": pod_enc, "dec": pod_dec},
        "CoLoRA": {"inner": stat_dod, "inner_coeff": stat_coeff, "coeff": colora_coeff}
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
    _, colora_residual, colora_sol = rom.evaluate_rom_forward(
                        'CoLoRA', fw['CoLoRA'], (mu_i, nu_i), sol, G
                    )
    colora_sol_vec = fom.solution_space.from_numpy(colora_sol)
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
    fom.visualize((u_i, colora_sol_vec, u_i - colora_sol_vec),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'CoLoRA-DL-ROM', f"Relative L\u00b2 error: mean={colora_residual:.3e}"))