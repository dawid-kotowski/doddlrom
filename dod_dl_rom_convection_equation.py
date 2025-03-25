from pymor.basic import *
import numpy as np

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
    dirichlet_data=ConstantFunction(dim_domain=2, value=0.),
    advection=advection_generic_function,
    name='advection_problem'
)

# Define the instationary problem
problem = InstationaryProblem(
    T=1.,
    initial_data=ExpressionFunction('20 * (x[0] < 1e-10)', 2),
    stationary_part=stationary_problem,
    name='advection_problem'
)

# Discretize the problem
fom, fom_data = discretize_instationary_cg(problem, diameter=0.1, nt=100)

# Define the parameter space with ranges for 'mu' and 'nu'
parameter_space = fom.parameters.space({'nu': (0, 10), 'mu': (0.2, np.pi-0.2)})

# Generate training and validation sets
training_set = parameter_space.sample_uniformly(2)
solution_set = fom.solution_space.empty()

# Create an empty list to hold the training data
training_data = []

# Solve the full-order model for each parameter in the training set
for mu_nu in training_set:
    solution = fom.solve(mu_nu)
    fom.visualize(solution)
    solution_set.append(solution)
    solution_flat = solution.to_numpy().flatten()
    training_data.append((mu_nu['mu'], mu_nu['nu'], solution_flat))

