from pymor.basic import *
from core import reduced_order_models as rom
from core.analytic.project import projector_check
from utils.paths import training_data_path, state_dicts_path
from core.configs.parameters import Ex02Parameters
from utils.paths import training_data_path, benchmarks_path
from utils.visualizer import vis_dl_diff, vis_dod_diff, vis_colora
import numpy as np
import torch

Nsample = 40

#region --- Configure this run ------------------------------------------------------
example_name = 'ex02'
P = Ex02Parameters(profile="baseline")          
P.assert_consistent()

#region --- Set up of FOM for pymor utility------------------------------------------
data = np.load(training_data_path(example_name) / f'full_order_training_data_{example_name}.npz')
mu = data['mu']          # shape [Ns, p]
nu = data['nu']          # shape [Ns, q]
solution = data['solution']  # shape [Ns, P.Nt, Nh]
Ns = mu.shape[0]
idx = np.arange(Ns)
np.random.shuffle(idx)
sel = idx[:Nsample]
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
G = np.load(training_data_path(example_name) / f'gram_matrix_{example_name}.npz')['gram'].astype(np.float32)


entry = training_data[0]
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


# ------------------ Load Modules --------------------------------
sd_dir = state_dicts_path(example_name)

def load_sd(m, p):
    m.load_state_dict(torch.load(p, map_location=device))
    m.eval()
    return m

# inner DOD used by DOD+DFNN and DOD-DL-ROM
innerDOD_model = rom.innerDOD(**P.make_innerDOD_kwargs()).to(device)

innerDOD_model = load_sd(innerDOD_model, sd_dir / "DOD_Module.pth")

# DOD+DFNN (DFNN -> N')
dfnn_nprime = rom.DFNN(**P.make_dod_dfnn_DFNN_kwargs()).to(device)

if (sd_dir / "DODFNN_Module.pth").exists():
    dfnn_nprime = load_sd(dfnn_nprime, sd_dir / "DODFNN_Module.pth")
else:
    raise FileNotFoundError("DOD+DFNN weights not found.")

# DOD-DL-ROM (DFNN -> n, AE: N'<->n)
dod_coeff = rom.DFNN(**P.make_dod_dl_DFNN_kwargs()).to(device)
dod_enc = rom.Encoder(**P.make_dod_dl_Encoder_kwargs()).to(device)
dod_dec = rom.Decoder(**P.make_dod_dl_Decoder_kwargs()).to(device)

if (sd_dir / "DOD_DL_ROM_Module.pth").exists():
    blob = torch.load(sd_dir / "DOD_DL_ROM_Module.pth", map_location=device)
    dod_enc.load_state_dict(blob["encoder"]);   dod_enc.eval()
    dod_dec.load_state_dict(blob["decoder"]);   dod_dec.eval()
    dod_coeff.load_state_dict(blob["coeff_model"]); dod_coeff.eval()
else:
    raise FileNotFoundError("DOD-DL-ROM weights not found.")

# POD-DL-ROM (DFNN -> n, AE: N_A<->n)
pod_coeff = rom.DFNN(**P.make_pod_DFNN_kwargs()).to(device)
pod_enc = rom.Encoder(**P.make_pod_Encoder_kwargs()).to(device)
pod_dec = rom.Decoder(**P.make_pod_Decoder_kwargs()).to(device)

if (sd_dir / "POD_DL_ROM_Module.pth").exists():
    blob = torch.load(sd_dir / "POD_DL_ROM_Module.pth", map_location=device)
    pod_enc.load_state_dict(blob["encoder"]); pod_enc.eval()
    pod_dec.load_state_dict(blob["decoder"]); pod_dec.eval()
    pod_coeff.load_state_dict(blob["coeff_model"]); pod_coeff.eval()
else:
    raise FileNotFoundError("POD-DL-ROM weights not found.")

# CoLoRA (stat -> N, CoLoRA N -> N, POD N -> N_h)
stat_dod = rom.statDOD(**P.make_statDOD_kwargs()).to(device)
stat_coeff = rom.statHadamardNN(**P.make_statHadamard_kwargs()).to(device)
colora_coeff = rom.CoLoRA(**P.make_CoLoRA_kwargs()).to(device)
if (sd_dir / "stat_DOD_Module.pth").exists():
    stat_dod = load_sd(stat_dod, sd_dir / "stat_DOD_Module.pth")
if (sd_dir / "stat_CoeffDOD_Module.pth").exists():
    stat_coeff = load_sd(stat_coeff, sd_dir  / "stat_CoeffDOD_Module.pth")
if (sd_dir / "CoLoRA_Module.pth").exists():
    colora_coeff = load_sd(colora_coeff, sd_dir / "CoLoRA_Module.pth")

torch.set_grad_enabled(False)

models = {
    "DOD+DFNN": {"inner": innerDOD_model, "coeff": dfnn_nprime},
    "DOD-DL-ROM": {"inner": innerDOD_model, "coeff": dod_coeff, "enc": dod_enc, "dec": dod_dec},
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


# --------- Check for Projection loss ---------------------------
projector_check(training_data, G, example_name, device, 'A_P')
projector_check(training_data, G, example_name, device, 'A')
projector_check(training_data, G, example_name, device, 'DOD', innerDOD_model=innerDOD_model)


# --------- Visualize -------------------------------------------
mu_list = mu_i.cpu().numpy().flatten().tolist()
nu_list = nu_i.cpu().numpy().flatten().tolist()

vis_dl_diff(fom, u_i, pod_dl_sol_vec, dod_dl_sol_vec, mu_list, nu_list)
vis_dod_diff(fom, u_i, dod_dfnn_sol_vec, dod_dl_sol_vec, mu_list, nu_list)
vis_colora(fom, u_i, colora_sol_vec, mu_list, nu_list)

# --------- Evaluate Error --------------------------------------
pod_split = rom.pod_dl_error_decomposition(
    fw['POD-DL-ROM'], training_data, example_name, G
)
print("[POD-DL-ROM | error decomposition]", pod_split)
dod_split = rom.dod_dl_error_decomposition(
    innerDOD_model, fw['DOD-DL-ROM'], training_data, P.N_prime, example_name, G
)
print("[DOD-DL-ROM | error decomposition]", dod_split)

csv_pod = str(benchmarks_path(example_name) / "error_decomp_pod.csv")
csv_dod = str(benchmarks_path(example_name) / "error_decomp_dod.csv")

wrote_pod = rom.append_error_decomp_csv(
    csv_pod, example_name, "POD-DL-ROM", pod_split, P, Nsample, dedup=False)
wrote_dod = rom.append_error_decomp_csv(
    csv_dod, example_name, "DOD-DL-ROM", dod_split, P, Nsample, dedup=False)

print("[POD-DL-ROM] has wrote to cvs: ", wrote_pod)
print("[DOD-DL-ROM] has wrote to cvs: ", wrote_dod)
