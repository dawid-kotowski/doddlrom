from pymor.basic import *
import numpy as np

# Constants
N_h = 20201
N_A = 64
nt = 10
diameter = 0.01
time_end = 1
sample_size = 400

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
    domain=RectDomain(left='neumann', right='neumann', top='neumann', bottom='neumann'),
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
fom, fom_data = discretize_instationary_cg(problem, diameter=diameter, nt=nt)

print(fom.parameters)

# Define the parameter space with ranges for 'mu' and 'nu'
parameter_space = fom.parameters.space({
    'mu': (0.3, 0.7),
    'nu': (0.002, 0.005)
})

# Generate training and validation sets
sample_size_per_param = int(np.ceil(np.power(sample_size, 1 / 4)))
training_set = parameter_space.sample_uniformly(sample_size_per_param)

# Create an empty list to hold the training data
training_data = []
solution_set = fom.solution_space.empty()


# Solve the full-order model for each parameter in the training set
i = 0
for mu_nu in training_set:
    solution = fom.solve(mu_nu)
    if i == int(len(training_set)/2):
        fom.visualize(solution)
    solution_set.append(solution)
    solution_flat = solution.to_numpy()
    training_data.append((mu_nu['mu'], mu_nu['nu'], solution_flat))
    i += 1


# Convert the training data list to a structured numpy array
dtype = [('mu', 'O'), ('nu', 'O'), ('solution', 'O')]
training_data_array = np.array(training_data, dtype=dtype)

# Save the numpy array to a file
np.save('examples/ex02/training_data/training_data_ex02.npy', training_data_array)

# Calculate the POD according to the full data
pod_modes, singular_values = pod(solution_set, product=fom.h1_0_semi_product, modes=N_A)
A = pod_modes.to_numpy().T

# Save POD to file
ambient = np.array(A, dtype=np.float32)
np.save('examples/ex02/training_data/ambient_matrix_ex02.npy', ambient)

# Calculate the Gram matrix for the fom
G = fom.h1_0_semi_product.matrix.toarray()

# Save G to file
gram = np.array(G, dtype=np.float32)
np.save('examples/ex02/training_data/gram_matrix_ex02.npy', gram)

# Save reduced training data to file
reduced_training_data = []
for mu_nu in training_set:
    solution = fom.solve(mu_nu)
    solution_flat = solution.to_numpy()
    reduced_solution = solution_flat @ G @ A
    reduced_training_data.append((mu_nu['mu'], mu_nu['nu'], reduced_solution))
reduced_training_data_array = np.array(reduced_training_data, dtype=dtype)
np.save('examples/ex02/training_data/reduced_training_data_ex02.npy', reduced_training_data_array)

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
np.save("examples/ex02/training_data/Dirchilet_shift_ex02.npy", u_0.to_numpy())

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
np.save('examples/ex02/training_data/stationary_training_data_ex02.npy', stat_training_data_array)

# Calculate the POD according to the full data
stat_pod_modes, stat_singular_values = pod(stat_solution_set, product=stat_fom.h1_0_semi_product, modes=N_A)
stat_A = stat_pod_modes.to_numpy().T

# Save POD to file
stat_ambient = np.array(stat_A, dtype=np.float32)
np.save('examples/ex02/training_data/stationary_ambient_matrix_ex02.npy', stat_ambient)

# Calculate the Gram matrix for the fom
stat_G = stat_fom.h1_0_semi_product.matrix.toarray()

# Save G to file
stat_gram = np.array(stat_G, dtype=np.float32)
np.save('examples/ex02/training_data/stationary_gram_matrix_ex02.npy', stat_gram)

# Save reduced training data to file
reduced_stationary_training_data = []
for mu_nu in training_set:
    stat_solution = stat_fom.solve(mu_nu)
    stat_solution_flat = stat_solution.to_numpy().flatten()
    reduced_stationary_solution = stat_solution_flat @ stat_G @ stat_A
    reduced_stationary_training_data.append((mu_nu['mu'], mu_nu['nu'], reduced_stationary_solution))
reduced_stationary_training_data_array = np.array(reduced_stationary_training_data, dtype=dtype)
np.save('examples/ex02/training_data/reduced_stationary_training_data_ex02.npy', reduced_stationary_training_data_array)
print(stat_solution_flat.shape)