from pymor.basic import *
from dod_dl_rom import DOD_DL, Coeff_DOD_DL
import numpy as np
import torch

# Usage example
N_h = 113
N_A = 10
rank = 10
L = 1
n = 4
m = 4
parameter_mu_dim = 1
parameter_nu_dim = 1
preprocess_dim = 113
nt = 10

# Initialize the models
DOD_DL_model = DOD_DL(1, parameter_mu_dim, [20, 10], n, N_A)
Coeff_model = Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, [10, 5])

# Load state_dicts
DOD_DL_model.load_state_dict(torch.load('DOD_Module.pth'))
DOD_DL_model.eval()
Coeff_model.load_state_dict(torch.load('DOD_Coefficient_Module.pth'))
Coeff_model.eval()

# Get some Validation Data
loaded_data = np.load('training_data.npy', allow_pickle=True)
# np.random.shuffle(loaded_data) # somehow shuffles all arrays ?
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
    dirichlet_data=ConstantFunction(dim_domain=2, value=0.),
    advection=advection_generic_function,
    name='advection_problem'
)
problem = InstationaryProblem(
    T=1.,
    initial_data=ExpressionFunction('(-(x[1] - 0.5)**2 + 0.25) * (x[0] < 1e-10)', 2),
    stationary_part=stationary_problem,
    name='advection_problem'
)
fom, fom_data = discretize_instationary_cg(problem, diameter=0.15, nt=nt)

# Set up solutions
true_solution = fom.solution_space.empty()
for entry in training_data:
    mu_i = torch.tensor(entry['mu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    mu_i = mu_i.unsqueeze(0)
    nu_i = torch.tensor(entry['nu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    nu_i = nu_i.unsqueeze(0)

    # True solution
    u_i = fom.solution_space.from_numpy(entry['solution'])
    true_solution.append(u_i)

    # Load ambient
    A = torch.tensor(np.load('ambient_matrix.npy', allow_pickle=True), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

    # DOD solution
    dod_solution = []
    for j in range(nt + 1):
        time = torch.tensor(j / (nt + 1), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        time = time.unsqueeze(0).unsqueeze(1)
        dod_dl_output = DOD_DL_model(mu_i, time).squeeze(0).T
        coeff_output = Coeff_model(mu_i, nu_i, time)
        u_i_dod = torch.matmul(torch.matmul(A, dod_dl_output), coeff_output)
        dod_solution.append(u_i_dod)  # append each [N_h] vector
    # Stack along the time axis to get shape [nt+1, N_h]
    u_i_dod = torch.stack(dod_solution, dim=0)
    u_i_dod = u_i_dod.detach().numpy()
    u_i_dod = fom.solution_space.from_numpy(u_i_dod)

    # Visualize
    fom.visualize((u_i, u_i_dod),
                  legend=(f'True solution for mu:{mu_i}, nu:{nu_i}',
                          f'Deep Orthogonal Decomposition mu:{mu_i}, nu:{nu_i}'))
