from pymor.basic import *
import numpy as np

# Constants
N_A = 64
nt = 10
diameter = 0.08

# Define the advection function dependent on 'mu'
def advection_function(x, mu):
    mu_value = mu['mu']
    return np.array([[np.cos(mu_value)*30, np.sin(mu_value)*30] for _ in range(x.shape[0])])

# Define the stationary problem
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

# Define the instationary problem
problem = InstationaryProblem(
    T=1.,
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
fom, fom_data = discretize_instationary_cg(problem, diameter=diameter, nt=nt)

# Define the parameter space with ranges for 'mu' and 'nu'
parameter_space = fom.parameters.space({'nu': (0.5, 1), 'mu': (0.2, np.pi-0.2)})

# Generate training and validation sets
training_set = parameter_space.sample_uniformly(30)

# Create an empty list to hold the training data
training_data = []
solution_set = fom.solution_space.empty()


# Solve the full-order model for each parameter in the training set
for mu_nu in training_set:
    solution = fom.solve(mu_nu)
    solution_set.append(solution)
    solution_flat = solution.to_numpy()
    training_data.append((mu_nu['mu'], mu_nu['nu'], solution_flat))
fom.visualize(solution)

# Convert the training data list to a structured numpy array
dtype = [('mu', 'O'), ('nu', 'O'), ('solution', 'O')]
training_data_array = np.array(training_data, dtype=dtype)

# Save the numpy array to a file
np.save('examples/ex01/training_data/training_data_ex01.npy', training_data_array)

# Calculate the POD according to the full data
pod_modes, singular_values = pod(solution_set, product=fom.h1_0_semi_product, modes=N_A)
A = pod_modes.to_numpy().T

# Save POD to file
ambient = np.array(A, dtype=np.float32)
np.save('examples/ex01/training_data/ambient_matrix_ex01.npy', ambient)

# Calculate the Gram matrix for the fom
G = fom.h1_0_semi_product.matrix.toarray()

# Save G to file
gram = np.array(G, dtype=np.float32)
np.save('examples/ex01/training_data/gram_matrix_ex01.npy', gram)


'''
----------------------------
Stationary Data for the CoLoRA inspired NN
----------------------------
'''

# Discretize the stationary problem
stat_fom, stat_fom_data = discretize_stationary_cg(stationary_problem, diameter=diameter)

# Define the parameter space with ranges for 'mu' and 'nu'
stat_parameter_space = stat_fom.parameters.space({'nu': (0.5, 1), 'mu': (0.2, np.pi-0.2)})

# Create an empty list for the stationary training data
stat_training_data = []
stat_solution_set = stat_fom.solution_space.empty()

# Correction for Dirichlet and save
u_0 = stat_fom.solve(stat_parameter_space.sample_uniformly(1)[0])
np.save("examples/ex01/training_data/Dirchilet_shift_ex01.npy", u_0.to_numpy())

# Solve the stationary full-order-model for each parameter
for mu_nu in training_set:
    stat_solution = stat_fom.solve(mu_nu)
    stat_solution_set.append(stat_solution - u_0)
    stat_solution_flat = stat_solution.to_numpy().flatten()
    stat_training_data.append((mu_nu['mu'], mu_nu['nu'], stat_solution_flat))
stat_fom.visualize(stat_solution)

# Convert the training data list to a structured numpy array
dtype = [('mu', 'O'), ('nu', 'O'), ('solution', 'O')]
stat_training_data_array = np.array(stat_training_data, dtype=dtype)

# Save the numpy array to a file
np.save('examples/ex01/training_data/stationary_training_data_ex01.npy', stat_training_data_array)

# Calculate the POD according to the full data
stat_pod_modes, stat_singular_values = pod(stat_solution_set, product=stat_fom.h1_0_semi_product, modes=N_A)
stat_A = stat_pod_modes.to_numpy().T

# Save POD to file
stat_ambient = np.array(stat_A, dtype=np.float32)
np.save('examples/ex01/training_data/stationary_ambient_matrix_ex01.npy', stat_ambient)

# Calculate the Gram matrix for the fom
stat_G = stat_fom.h1_0_semi_product.matrix.toarray()

# Save G to file
stat_gram = np.array(stat_G, dtype=np.float32)
np.save('examples/ex01/training_data/stationary_gram_matrix_ex01.npy', stat_gram)