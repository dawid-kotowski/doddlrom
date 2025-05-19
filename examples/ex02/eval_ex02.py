from pymor.basic import *
from master_project_1 import dod_dl_rom as dr
import numpy as np
import torch

# Usage example
N_h = 20201
N_A = 64
nt = 10
diameter = 0.01
L = 3
N = 16
n = 4
m = 4
parameter_mu_dim = 3
parameter_nu_dim = 1
preprocess_dim = 2
dod_structure = [64, 64]
phi_n_structure = [16, 8]
coeff_ae_structure = [32, 16, 8]
stat_dod_structure = [128, 64]
pod_in_channels = 1
pod_hidden_channels = 1
lin_dim_ae = 0
kernel = 3
stride = 2
padding = 1

#region Loading of all Models
# Initialize the models
DOD_DL_model = dr.DOD_DL(preprocess_dim, parameter_mu_dim, dod_structure, n, N_A)
DOD_DL_coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)

output = int(np.sqrt(N_A))
pod_num_layers = 0
while (output - int(np.sqrt(n)) > lin_dim_ae):
    output = int(np.floor((output + 2*padding - kernel) / stride) + 1)
    pod_num_layers += 1
Decoder_model = dr.Decoder(N_A, pod_in_channels, pod_hidden_channels, n, pod_num_layers, kernel, stride, padding)
POD_DL_coeff_model = dr.Coeff_AE(parameter_mu_dim, parameter_nu_dim, n, coeff_ae_structure)

stat_DOD_model = dr.DOD(parameter_mu_dim, preprocess_dim, n, N_A, stat_dod_structure)
stat_Coeff_model = dr.CoeffDOD(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)
CoLoRA_DL_model = dr.CoLoRA_DL(N_A, L, n, parameter_nu_dim)

# Load state_dicts
DOD_DL_model.load_state_dict(torch.load('examples/ex02/state_dicts/DOD_Module.pth'))
DOD_DL_model.eval()
DOD_DL_coeff_model.load_state_dict(torch.load('examples/ex02/state_dicts/DOD_Coefficient_Module.pth'))
DOD_DL_coeff_model.eval()
checkpoint = torch.load('examples/ex02/state_dicts/POD_DL_Module.pth')

Decoder_model.load_state_dict(checkpoint['decoder'])
POD_DL_coeff_model.load_state_dict(checkpoint['coeff_model'])
Decoder_model.eval()
POD_DL_coeff_model.eval()

stat_DOD_model.load_state_dict(torch.load('examples/ex02/state_dicts/stat_DOD_Module.pth'))
stat_Coeff_model.load_state_dict(torch.load('examples/ex02/state_dicts/stat_CoeffDOD_Module.pth'))
stat_DOD_model.eval()
stat_Coeff_model.eval()
CoLoRA_DL_model.load_state_dict(torch.load('examples/ex02/state_dicts/CoLoRA_Module.pth'))
CoLoRA_DL_model.eval()

#endregion

#region Set up of FOM for pymor utility
# Get some Validation Data
loaded_data = np.load('examples/ex02/training_data/training_data_ex02.npy', allow_pickle=True)
np.random.shuffle(loaded_data) 
training_data = loaded_data[:4]

# Define Full Order Model again
def advection_function(x, mu):
    mu_value = mu['mu']
    return np.array([[np.cos(mu_value)*30, np.sin(mu_value)*30] for _ in range(x.shape[0])])
advection_params = Parameters({'mu': 1})
advection_generic_function = GenericFunction(advection_function, dim_domain=2, shape_range=(2,), parameters=advection_params)
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
fom, fom_data = discretize_instationary_cg(problem, diameter=diameter, nt=nt)

#endregion

# Set up solutions
true_solution = fom.solution_space.empty()
A = torch.tensor(np.load('examples/ex02/training_data/ambient_matrix_ex02.npy', allow_pickle=True), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

for entry in training_data:
    mu_i = torch.tensor(entry['mu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    mu_i = mu_i.unsqueeze(0)
    nu_i = torch.tensor(entry['nu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    nu_i = nu_i.unsqueeze(0)

    # True solution
    u_i = fom.solution_space.from_numpy(entry['solution'])
    true_solution.append(u_i)

    # Coeff_DL + POD_DL + CoLoRA_DL solution
    pod_dl_sol = fom.solution_space.from_numpy(
        dr.pod_dl_forward(A, POD_DL_coeff_model, Decoder_model, mu_i, nu_i, nt))
    u_i_colora = fom.solution_space.from_numpy(
        dr.colora_dl_forward(A, stat_DOD_model, stat_Coeff_model, CoLoRA_DL_model, 
                             mu_i, nu_i, nt))
    coeff_dl_sol = fom.solution_space.from_numpy(
        dr.dod_dl_forward(A, DOD_DL_model, DOD_DL_coeff_model, mu_i, nu_i, nt))

    # Visualize
    fom.visualize((u_i, coeff_dl_sol, pod_dl_sol, u_i_colora),
                  legend=(f'True solution for mu:{mu_i}, nu:{nu_i}',
                          f'Linear Coefficient DOD-DL-ROM for mu:{mu_i}, nu:{nu_i}',
                          f'POD-DL-ROM for mu{mu_i}, nu:{nu_i}',
                          f'DOD pretrained CoLoRA for mu:{mu_i}, nu:{nu_i}'))
