from pymor.basic import *
from core import reduced_order_models as rom
from core.analytic.project import projector_check
from utils.paths import training_data_path, state_dicts_path
from core.configs.parameters import Ex04Parameters 
from utils.paths import training_data_path, benchmarks_path
import numpy as np
import torch

Nsample = 1

#region --- Configure this run ------------------------------------------------------
example_name = 'ex04'
P = Ex04Parameters(profile="baseline")          
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
from core.bindings.fom import discretize

default_config = {
    "reduction": 1e-20,
    "grid.dim": 2,
    "grid.yasp_x": P.grid_size,
    "grid.yasp_y": P.grid_size,
    "time.time": 0.0,
    "time.dt": P.dt,
    "time.solverSteps": 0.01,
    "time.T": P.T,
    "problem.eta": 0.2,
    "problem.inflowVelocity" : 1.0,
    "problem.non-parametric.openingHeight": 0.3,
    "problem.parametric.coatingHeight": 0.0,
    "problem.parametric.minPermeability": 0.0,
    "problem.parametric.coatingPermeability": 0.0,
    "problem.parametric.inflowAngle": 0.0,
    "darcy.reduction": 1e-12,
    "visualization.subsampling": 8,
    "visualization.subsamplingVelocity": 5,
    "visualization.subsamplingDG": 5,
}

fom = discretize(default_config)
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

torch.set_grad_enabled(False)

models = {
    "DOD+DFNN": {"inner": innerDOD_model, "coeff": dfnn_nprime},
    "DOD-DL-ROM": {"inner": innerDOD_model, "coeff": dod_coeff, "enc": dod_enc, "dec": dod_dec},
    "POD-DL-ROM": {"coeff": pod_coeff, "enc": pod_enc, "dec": pod_dec}
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
_, dod_dfnn_residual, dod_dfnn_sol = rom.evaluate_rom_forward(
                    'DOD+DFNN', fw['DOD+DFNN'], (mu_i, nu_i), sol, G
                )
dod_dfnn_sol_vec = fom.solution_space.from_numpy(dod_dfnn_sol)


# --------- Check for Projection loss ---------------------------
projector_check(training_data, G, example_name, device, 'A_P')
projector_check(training_data, G, example_name, device, 'A')
projector_check(training_data, G, example_name, device, 'DOD', innerDOD_model=innerDOD_model)


# --------- Visualize -------------------------------------------
path = benchmarks_path("ex04")
mu_list = mu_i.cpu().numpy().flatten().tolist()
nu_list = nu_i.cpu().numpy().flatten().tolist()

print("|------------- Mu values -------------|")
print("|", mu_list, "|")
print("|------------- Nu values -------------|")
print("|", mu_list, "|")
print("|-------------------------------------|")
fom.visualize(u_i, filename=path / "true_solution")
fom.visualize(pod_dl_sol_vec, filename=path / "pod_dl_rom_solution")
fom.visualize(dod_dl_sol_vec, filename=path / "dod_dl_rom_solution")
fom.visualize(dod_dfnn_sol_vec, filename=path / "dod_dfnn_solution")


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
