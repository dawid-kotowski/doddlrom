from pymor.basic import *
from master_project_1 import reduced_order_models as dr
import numpy as np
import torch

# Usage example
N_h = 5101
N_A = 64
N = 64
N_prime = 16
n = 4
Nt = 10
diameter = 0.02
parameter_mu_dim = 1
parameter_nu_dim = 1
# DOD+DFNN
preprocess_dim = 2
dod_structure = [32, 16]
df_layers = [16, 8]
# DOD-DL-ROM
dod_dl_df_layers = [32, 16, 8]
dod_in_channels = 1
dod_hidden_channels = 1
dod_lin_dim_ae = 0
dod_kernel = 3
dod_stride = 2
dod_padding = 1
# POD-DL-ROM
pod_df_layers = [32, 16, 8]
pod_in_channels = 1
pod_hidden_channels = 1
pod_lin_dim_ae = 0
pod_kernel = 3
pod_stride = 2
pod_padding = 1
# CoLoRA-ROM
L = 3
stat_m = 4
stat_dod_structure = [128, 64]
stat_phi_n_structure = [16, 8]

# Training Example
generalepochs = 500
generalrestarts = 10
generalpatience = 3

#region Loading of all Models
# Initialize the models
innerDOD_model = dr.innerDOD(preprocess_dim, parameter_mu_dim, dod_structure, N_prime, N_A)
DFNN_Nprime_model = dr.DFNN(parameter_mu_dim, parameter_nu_dim, N_prime, df_layers)

output = int(np.sqrt(N_A))
pod_num_layers = 0
while (output - int(np.sqrt(n)) > pod_lin_dim_ae):
    output = int(np.floor((output + 2*pod_padding - pod_kernel) / pod_stride) + 1)
    pod_num_layers += 1
Decoder_model = dr.Decoder(N, pod_in_channels, pod_hidden_channels, 
                           n, pod_num_layers, pod_kernel, pod_stride, pod_padding)
DFNN_POD_n_model = dr.DFNN(parameter_mu_dim, parameter_nu_dim, n, pod_df_layers)

output = int(np.sqrt(N_prime))
dod_num_layers = 0
while (output - int(np.sqrt(n)) > dod_lin_dim_ae):
    output = int(np.floor((output + 2*dod_padding - dod_kernel) / dod_stride) + 1)
    dod_num_layers += 1
DOD_Decoder_model = dr.Decoder(N_prime, dod_in_channels, dod_hidden_channels, n, 
                      dod_num_layers, pod_kernel, dod_stride, dod_padding)
DFNN_DOD_n_model = dr.DFNN(parameter_mu_dim, parameter_nu_dim, n, dod_dl_df_layers)

stat_DOD_model = dr.statDOD(parameter_mu_dim, preprocess_dim, N_prime, N_A, stat_dod_structure)
stat_Coeff_model = dr.statHadamardNN(parameter_mu_dim, parameter_nu_dim, stat_m, 
                               N_prime, stat_phi_n_structure)
CoLoRA_model = dr.CoLoRA(N_A, L, N_prime, parameter_nu_dim)

# Load state_dicts
innerDOD_model.load_state_dict(torch.load('examples/ex01/state_dicts/DOD_Module.pth'))
innerDOD_model.eval()

DFNN_Nprime_model.load_state_dict(torch.load('examples/ex01/state_dicts/DODFNN_Module.pth'))
DFNN_Nprime_model.eval()

checkpoint = torch.load('examples/ex01/state_dicts/DOD_DL_ROM_Module.pth')
DOD_Decoder_model.load_state_dict(checkpoint['decoder'])
DFNN_DOD_n_model.load_state_dict(checkpoint['coeff_model'])
DOD_Decoder_model.eval()
DFNN_DOD_n_model.eval()

checkpoint = torch.load('examples/ex01/state_dicts/POD_DL_ROM_Module.pth')
Decoder_model.load_state_dict(checkpoint['decoder'])
DFNN_POD_n_model.load_state_dict(checkpoint['coeff_model'])
Decoder_model.eval()
DFNN_POD_n_model.eval()

stat_DOD_model.load_state_dict(torch.load('examples/ex01/state_dicts/stat_DOD_Module.pth'))
stat_Coeff_model.load_state_dict(torch.load('examples/ex01/state_dicts/stat_CoeffDOD_Module.pth'))
stat_DOD_model.eval()
stat_Coeff_model.eval()
CoLoRA_model.load_state_dict(torch.load('examples/ex01/state_dicts/CoLoRA_Module.pth'))
CoLoRA_model.eval()

#endregion

#region Set up of FOM for pymor utility
data = np.load('examples/ex01/training_data/full_order_training_data_ex01.npz')
mu = data['mu']          # shape [Ns]
nu = data['nu']          # shape [Ns]
solution = data['solution']  # shape [Ns, Nt, Nh]
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
fom, fom_data = discretize_instationary_cg(problem, diameter=diameter, nt=Nt)

#endregion

# Set up solutions
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
    pod_dl_sol = dr.pod_dl_rom_forward(A_P, DFNN_POD_n_model, 
                                       Decoder_model, mu_i, nu_i, Nt)  # -> [Nt+1, Nh]
    Tmin = min(sol.shape[0], pod_dl_sol.shape[0])
    pod_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - pod_dl_sol[:Tmin]))
    pod_dl_sol_vec = fom.solution_space.from_numpy(pod_dl_sol)

    # --- DOD-DL-ROM ---
    dod_dl_sol = dr.dod_dl_rom_forward(A, innerDOD_model, DFNN_DOD_n_model, 
                                       DOD_Decoder_model, mu_i, nu_i, Nt)  # -> [Nt+1, Nh]
    Tmin = min(sol.shape[0], dod_dl_sol.shape[0])
    dod_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - dod_dl_sol[:Tmin]))
    dod_dl_sol_vec = fom.solution_space.from_numpy(dod_dl_sol)

    # --- CoLoRA ---
    u_i_colora = dr.colora_forward(A, stat_DOD_model, stat_Coeff_model, 
                                   CoLoRA_model, mu_i, nu_i, Nt)  # -> [Nt+1, Nh]
    Tmin = min(sol.shape[0], u_i_colora.shape[0])
    colora_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - u_i_colora[:Tmin]))
    u_i_colora_vec = fom.solution_space.from_numpy(u_i_colora)

    # --- DOD+DFNN ---
    coeff_dl_sol = dr.dod_dfnn_forward(A, innerDOD_model, DFNN_Nprime_model, 
                                         mu_i, nu_i, Nt)  # -> [Nt+1, Nh]
    Tmin = min(sol.shape[0], coeff_dl_sol.shape[0])
    dod_dl_residual = fom.solution_space.from_numpy(np.abs(sol[:Tmin] - coeff_dl_sol[:Tmin]))
    coeff_dl_sol_vec = fom.solution_space.from_numpy(coeff_dl_sol)


    # Visualize
    fom.visualize((u_i, coeff_dl_sol_vec, dod_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()}, ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'linear DOD-DL-ROM', 'L² - Error'))
    fom.visualize((u_i, pod_dl_sol_vec, pod_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'POD-DL-ROM', 'L² - Error'))
    fom.visualize((u_i, dod_dl_sol_vec, dod_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'DOD-DL-ROM', 'L² - Error'))
    fom.visualize((u_i, u_i_colora_vec, colora_dl_residual),
                  legend=(f'FOM for μ = {mu_i.cpu().numpy().flatten().tolist()},  ν = {nu_i.cpu().numpy().flatten().tolist()}', 
                          'CoLoRA-DL-ROM', 'L² - Error'))