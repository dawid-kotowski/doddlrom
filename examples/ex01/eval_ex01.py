from pymor.basic import *
from master_project_1 import dod_dl_rom as dr
import numpy as np
import torch

# Usage example
N_h = 5101
N_A = 64
rank = 10
L = 3
N = 16
n = 4
m = 4
parameter_mu_dim = 1
parameter_nu_dim = 1
preprocess_dim = 2
dod_structure = [128, 64]
phi_N_structure = [32, 16]
phi_n_structure = [16, 8]
stat_dod_structure = [128, 64]
nt = 10
diameter = 0.02

# Initialize the models
DOD_DL_model = dr.DOD_DL(preprocess_dim, parameter_mu_dim, dod_structure, N, N_A)
Coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, N, phi_N_structure)

Decoder_model = dr.Decoder(N, 1, 1, n, 1, kernel=3, stride=2, padding=1)
AE_Coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)

stat_DOD_model = dr.DOD(preprocess_dim, n, N_A, stat_dod_structure)
stat_Coeff_model = dr.CoeffDOD(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)
CoLoRA_DL_model = dr.CoLoRA_DL(N_A, L, n, parameter_nu_dim)

# Load state_dicts
DOD_DL_model.load_state_dict(torch.load('examples/ex01/state_dicts/DOD_Module.pth'))
DOD_DL_model.eval()
Coeff_model.load_state_dict(torch.load('examples/ex01/state_dicts/DOD_Coefficient_Module.pth'))
Coeff_model.eval()
checkpoint = torch.load('examples/ex01/state_dicts/AE_DOD_DL_Module.pth')

Decoder_model.load_state_dict(checkpoint['decoder'])
AE_Coeff_model.load_state_dict(checkpoint['coeff_model'])
Decoder_model.eval()
AE_Coeff_model.eval()

stat_DOD_model.load_state_dict(torch.load('examples/ex01/state_dicts/stat_DOD_Module.pth'))
stat_Coeff_model.load_state_dict(torch.load('examples/ex01/state_dicts/stat_CoeffDOD_Module.pth'))
stat_DOD_model.eval()
stat_Coeff_model.eval()
CoLoRA_DL_model.load_state_dict(torch.load('examples/ex01/state_dicts/CoLoRA_Module.pth'))
CoLoRA_DL_model.eval()

# Get some Validation Data
loaded_data = np.load('examples/ex01/training_data/training_data_ex01.npy', allow_pickle=True)
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

# Set up solutions
true_solution = fom.solution_space.empty()
A = torch.tensor(np.load('examples/ex01/training_data/ambient_matrix_ex01.npy', allow_pickle=True), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

for entry in training_data:
    mu_i = torch.tensor(entry['mu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    mu_i = mu_i.unsqueeze(0)
    nu_i = torch.tensor(entry['nu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    nu_i = nu_i.unsqueeze(0)

    # True solution
    u_i = fom.solution_space.from_numpy(entry['solution'])
    true_solution.append(u_i)

    # Coeff_DL + AE_DOD_DL + CoLoRA_DL solution
    coeff_dl_solution = []
    ae_dl_solution = []
    colora_dl_solution = []
    for j in range(nt + 1):
        time = torch.tensor(j / (nt + 1), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        time = time.unsqueeze(0).unsqueeze(1)
        dod_dl_output = DOD_DL_model(mu_i, time).squeeze(0).T
        coeff_output = Coeff_model(mu_i, nu_i, time)
        u_i_coeff_dl = torch.matmul(torch.matmul(A, dod_dl_output), coeff_output)
        coeff_dl_solution.append(u_i_coeff_dl)  # append each [N_h] vector

        coeff_n_output = AE_Coeff_model(mu_i, nu_i, time).unsqueeze(0)
        decoded_output = Decoder_model(coeff_n_output).squeeze(0)
        u_i_ae_dl = torch.matmul(torch.matmul(A, dod_dl_output), decoded_output)
        ae_dl_solution.append(u_i_ae_dl)

        stat_coeff_n_output = stat_Coeff_model(mu_i, nu_i).unsqueeze(0).unsqueeze(2)
        stat_dod_output = stat_DOD_model(mu_i)
        v_0 = torch.bmm(stat_dod_output.transpose(1, 2), stat_coeff_n_output)
        u_i_colora = torch.matmul(A, CoLoRA_DL_model(v_0, nu_i, time).squeeze(0))
        colora_dl_solution.append(u_i_colora)

    # Stack along the time axis to get shape [nt+1, N_h]
    coeff_dl_sol = torch.stack(coeff_dl_solution, dim=0)
    coeff_dl_sol = coeff_dl_sol.detach().numpy()
    coeff_dl_sol = fom.solution_space.from_numpy(coeff_dl_sol)

    ae_dl_sol = torch.stack(ae_dl_solution, dim=0)
    ae_dl_sol = ae_dl_sol.detach().numpy()
    ae_dl_sol = fom.solution_space.from_numpy(ae_dl_sol)

    u_i_colora = torch.stack(colora_dl_solution, dim=0)
    u_i_colora = u_i_colora.detach().numpy()
    u_i_colora = fom.solution_space.from_numpy(u_i_colora)

    # Visualize
    fom.visualize((u_i, coeff_dl_sol, ae_dl_sol, u_i_colora),
                  legend=(f'True solution for mu:{mu_i}, nu:{nu_i}',
                          f'Linear Coefficient DOD-DL-ROM for mu:{mu_i}, nu:{nu_i}',
                          f'AE improved DOD-DL-ROM for mu{mu_i}, nu:{nu_i}',
                          f'DOD pretrained CoLoRA for mu:{mu_i}, nu:{nu_i}'))
