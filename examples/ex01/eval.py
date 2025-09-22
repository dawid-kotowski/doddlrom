from pymor.basic import *
from core import reduced_order_models as rom
from core.configs.parameters import Ex01Parameters  
import numpy as np
import torch

#region --- Configure this run ------------------------------------------------------
P = Ex01Parameters(profile="tiny")          # or "wide"/"tiny"/"debug"
P.assert_consistent()

# --- Build models ------------------------------------------------------------------

innerDOD_model = rom.innerDOD(**P.make_innerDOD_kwargs())

DFNN_Nprime_model = rom.DFNN(**P.make_dod_dfnn_DFNN_kwargs())

POD_coeff_model = rom.DFNN(**P.make_pod_DFNN_kwargs())
POD_Encoder_model = rom.Encoder(**P.make_pod_Encoder_kwargs())
POD_Decoder_model = rom.Decoder(**P.make_pod_Decoder_kwargs())

DOD_DL_coeff_model = rom.DFNN(**P.make_dod_dl_DFNN_kwargs())
DOD_DL_Encoder_model = rom.Encoder(**P.make_dod_dl_Encoder_kwargs())
DOD_DL_Decoder_model = rom.Decoder(**P.make_dod_dl_Decoder_kwargs())

stat_DOD_model = rom.statDOD(**P.make_statDOD_kwargs())
stat_Coeff_model = rom.statHadamardNN(**P.make_statHadamard_kwargs())
CoLoRA_model = rom.CoLoRA(**P.make_CoLoRA_kwargs())

# Load state_dicts
innerDOD_model.load_state_dict(torch.load('examples/ex01/state_dicts/DOD_Module.pth'))
innerDOD_model.eval()

DFNN_Nprime_model.load_state_dict(torch.load('examples/ex01/state_dicts/DODFNN_Module.pth'))
DFNN_Nprime_model.eval()

checkpoint = torch.load('examples/ex01/state_dicts/DOD_DL_ROM_Module.pth')
DOD_DL_Decoder_model.load_state_dict(checkpoint['decoder'])
DOD_DL_coeff_model.load_state_dict(checkpoint['coeff_model'])
DOD_DL_Decoder_model.eval()
DOD_DL_coeff_model.eval()

checkpoint = torch.load('examples/ex01/state_dicts/POD_DL_ROM_Module.pth')
POD_Decoder_model.load_state_dict(checkpoint['decoder'])
POD_coeff_model.load_state_dict(checkpoint['coeff_model'])
POD_Decoder_model.eval()
POD_coeff_model.eval()

stat_DOD_model.load_state_dict(torch.load('examples/ex01/state_dicts/stat_DOD_Module.pth'))
stat_Coeff_model.load_state_dict(torch.load('examples/ex01/state_dicts/stat_CoeffDOD_Module.pth'))
stat_DOD_model.eval()
stat_Coeff_model.eval()
CoLoRA_model.load_state_dict(torch.load('examples/ex01/state_dicts/CoLoRA_Module.pth'))
CoLoRA_model.eval()

#endregion

#region --- Set up of FOM for pymor utility------------------------------------------
data = np.load('examples/ex01/training_data/full_order_training_data_ex01.npz')
mu = data['mu']          # shape [Ns]
nu = data['nu']          # shape [Ns]
solution = data['solution']  # shape [Ns, P.Nt, Nh]
Ns = mu.shape[0]
idx = np.arange(Ns)
np.random.shuffle(idx)
sel = idx[:1]
training_data = [(float(mu[i]), float(nu[i]), solution[i]) for i in sel]

# Define Full Order Model again
def advection_function(x, mu):
    mu_value = mu['mu']
    return np.array([[np.cos(mu_value)*30, np.sin(mu_value)*30] for _ in range(x.shape[0])])
advection_params = Parameters({'mu': 1})
advection_generic_function = GenericFunction(advection_function, 
                                             dim_domain=2, shape_range=(2,), 
                                             parameters=advection_params)
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

A = torch.tensor(
    np.load('examples/ex01/training_data/N_A_ambient_ex01.npz')['ambient'],
    dtype=torch.float32, device=device
)
A_P = torch.tensor(
    np.load('examples/ex01/training_data/N_ambient_ex01.npz')['ambient'],
    dtype=torch.float32, device=device
)

for entry in training_data:
    # unpack tuple
    mu, nu, sol = entry

    # ensure shape [1,1] for concatenation in your models
    mu_i = torch.tensor([[mu]], dtype=torch.float32, device=device)
    nu_i = torch.tensor([[nu]], dtype=torch.float32, device=device)

    # True solution (VectorArray)
    u_i = fom.solution_space.from_numpy(sol)
    true_solution.append(u_i)

    # --- POD-DL-ROM ---
    pod_dl_sol = rom.pod_dl_rom_forward(A_P, POD_coeff_model, 
                                       POD_Decoder_model, mu_i, nu_i, P.Nt, 
                                       example_name='ex01', reduction_tag='N_reduced')  # -> [P.Nt+1, Nh]
    Tmin = min(sol.shape[0], pod_dl_sol.shape[0])
    pod_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - pod_dl_sol[:Tmin]))
    pod_dl_sol_vec = fom.solution_space.from_numpy(pod_dl_sol)

    # --- DOD-DL-ROM ---
    dod_dl_sol = rom.dod_dl_rom_forward(A, innerDOD_model, DOD_DL_coeff_model, 
                                       DOD_DL_Decoder_model, mu_i, nu_i, P.Nt, 
                                       example_name='ex01', reduction_tag='N_A_reduced')  # -> [P.Nt+1, Nh]
    Tmin = min(sol.shape[0], dod_dl_sol.shape[0])
    dod_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - dod_dl_sol[:Tmin]))
    dod_dl_sol_vec = fom.solution_space.from_numpy(dod_dl_sol)

    # --- CoLoRA ---
    u_i_colora = rom.colora_forward(A, stat_DOD_model, stat_Coeff_model, 
                                   CoLoRA_model, mu_i, nu_i, P.Nt, 
                                   example_name='ex01', reduction_tag='N_A_reduced')  # -> [P.Nt+1, Nh]
    Tmin = min(sol.shape[0], u_i_colora.shape[0])
    colora_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - u_i_colora[:Tmin]))
    u_i_colora_vec = fom.solution_space.from_numpy(u_i_colora)

    # --- DOD+DFNN ---
    coeff_dl_sol = rom.dod_dfnn_forward(A, innerDOD_model, DFNN_Nprime_model, 
                                         mu_i, nu_i, P.Nt, 
                                         example_name='ex01', reduction_tag='N_A_reduced')  # -> [P.Nt+1, Nh]
    Tmin = min(sol.shape[0], coeff_dl_sol.shape[0])
    dod_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - coeff_dl_sol[:Tmin]))
    coeff_dl_sol_vec = fom.solution_space.from_numpy(coeff_dl_sol)


    # Visualize
    fom.visualize((u_i, coeff_dl_sol_vec, dod_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()}, ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'DOD+DFNN', 'L² - Error'))
    fom.visualize((u_i, pod_dl_sol_vec, pod_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'POD-DL-ROM', 'L² - Error'))
    fom.visualize((u_i, dod_dl_sol_vec, dod_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'DOD-DL-ROM', 'L² - Error'))
    fom.visualize((u_i, u_i_colora_vec, colora_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'CoLoRA-DL-ROM', 'L² - Error'))