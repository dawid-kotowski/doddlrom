import torch
from master_project_1 import dod_dl_rom as dr
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.benchmark as benchmark
import pickle

# Fixed Constants
N_h = 5101
N_A = 64
nt = 10
diameter = 0.02
L = 3
N = 16
# unspecified n
m = 4
parameter_mu_dim = 1
parameter_nu_dim = 1
preprocess_dim = 2
dod_structure = [64, 64]
phi_n_structure = [16, 8]
coeff_ae_structure = [8, 4, 4]
stat_dod_structure = [128, 64]
pod_in_channels = 1
pod_hidden_channels = 1
pod_num_layers = 3
#Training Example
generalepochs = 500
generalrestarts = 10
generalpatience = 5

# Fetch Training and Validation set
train_valid_data = dr.FetchReducedTrainAndValidSet(0.8, 'ex01')
stat_train_valid_data = dr.StatFetchReducedTrainAndValidSet(0.8, 'ex01')

# Fetch Ambient and Gram Matrix
G = np.load('examples/ex01/training_data/gram_matrix_ex01.npy', allow_pickle=True)
A_np = np.load('examples/ex01/training_data/ambient_matrix_ex01.npy', allow_pickle=True)
A = torch.tensor(A_np, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize random Performance Check set
loaded_data = np.load('examples/ex01/training_data/training_data_ex01.npy', allow_pickle=True)
np.random.shuffle(loaded_data) 
performance_data = loaded_data[:5]

# Initialize Error Loader of Performance Check set
l2_norm = dr.L2Norm(np.load('examples/ex01/training_data/gram_matrix_ex01.npy', allow_pickle=True))
error_loader = dr.MeanError(l2_norm)

'''
-------------------------
-------------------------
Start of Performance Loop
-------------------------
-------------------------
'''

abs_error_lin_dod_dl = []
rel_error_lin_dod_dl = []
abs_error_pod_dl = []
rel_error_pod_dl = []
abs_error_colora_dl = []
rel_error_colora_dl = []
ambient_errors = []
time_results = []
for n in tqdm(range(2, 8), desc="Reduced Dimension"):

    #region Initialisation of all Models
    # Initialize the DOD model
    DOD_DL_model = dr.DOD_DL(preprocess_dim, parameter_mu_dim, dod_structure, n, N_A)

    # Initialize the DOD trainer
    DOD_DL_trainer = dr.DOD_DL_Trainer(DOD_DL_model, train_valid_data, N_A, 'ex01',
                                    generalepochs,generalrestarts, learning_rate=1e-3, 
                                    batch_size=128, patience=generalpatience)

    # Train the DOD model
    best_loss = DOD_DL_trainer.train()

    # Initialize the Coefficient Finding model
    DOD_DL_coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)

    # Initialize the Coefficient Finding trainer
    DOD_DL_coeff_trainer = dr.Coeff_DOD_DL_Trainer(N_A, DOD_DL_model, DOD_DL_coeff_model,
                                    train_valid_data, 'ex01', 
                                    generalepochs, generalrestarts, learning_rate=1e-3, 
                                    batch_size=128, patience=generalpatience)

    # Train the Coefficient model
    best_loss2 = DOD_DL_coeff_trainer.train()

    # Initialize the POD DL ROM model
    En_model = dr.Encoder(N_A, pod_in_channels, pod_hidden_channels, n, pod_num_layers, kernel=3, stride=2, padding=1)
    De_model = dr.Decoder(N_A, pod_in_channels, pod_hidden_channels, n, pod_num_layers, kernel=3, stride=2, padding=1)
    POD_DL_coeff_model = dr.Coeff_AE(parameter_mu_dim, parameter_nu_dim, n, coeff_ae_structure)

    # Initialize the AE Coefficient Finding trainer
    POD_DL_coeff_trainer = dr.POD_DL_Trainer(POD_DL_coeff_model, En_model, De_model,
                                        train_valid_data, 'ex01', 0.5,
                                        generalepochs, generalrestarts, learning_rate=1e-3, 
                                        batch_size=128, patience=generalpatience)

    # Train the AE Coefficient model
    best_loss3 = POD_DL_coeff_trainer.train()

    # Initialize and train the stationary DOD model
    stat_DOD_model = dr.DOD(preprocess_dim, n, N_A, stat_dod_structure)
    stat_DOD_Trainer = dr.DODTrainer(stat_DOD_model, N_A, 
                                    stat_train_valid_data, 'ex01', 
                                    generalepochs, generalrestarts, learning_rate=1e-3, 
                                    batch_size=128, patience=generalpatience)
    best_loss4 = stat_DOD_Trainer.train()

    # Initialize and train the stationary Coefficient Finding model
    stat_Coeff_model = dr.CoeffDOD(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)
    stat_Coeff_Trainer = dr.CoeffDODTrainer(stat_DOD_model, stat_Coeff_model, N_A,
                                            stat_train_valid_data, 'ex01',
                                            generalepochs, generalrestarts, learning_rate=1e-3, 
                                            batch_size=128, patience=generalpatience)
    best_loss5 = stat_Coeff_Trainer.train()

    # Initialize the CoLoRA_DL model
    CoLoRA_DL_model = dr.CoLoRA_DL(N_A, L, n, parameter_nu_dim)

    # Initialize the CoLoRA_DL trainer
    CoLoRa_DL_Trainer = dr.CoLoRA_DL_Trainer(N_A, stat_DOD_model, stat_Coeff_model, 
                                            CoLoRA_DL_model, train_valid_data, 'ex01',
                                            generalepochs, generalrestarts, learning_rate=1e-3, 
                                            batch_size=128, patience=generalpatience)

    # Train the CoLoRA_DL
    best_loss6 = CoLoRa_DL_Trainer.train()

    #endregion

    '''
    ---------------------
    Start of Evaluation on random set
    ---------------------
    '''

    # Set all to evalutate
    DOD_DL_model.eval()
    DOD_DL_coeff_model.eval()
    De_model.eval()
    POD_DL_coeff_model.eval()
    stat_DOD_model.eval()
    stat_Coeff_model.eval()
    CoLoRA_DL_model.eval()

    # Set up timers for some solution
    mu_0 = torch.tensor(performance_data[0]['mu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    mu_0 = mu_0.unsqueeze(0)
    nu_0 = torch.tensor(performance_data[0]['nu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    nu_0 = nu_0.unsqueeze(0)
    label = f'Forward Through DL-ROMs for Latent Dimension {n}'
    for num_threads in [1, 2, 4, 8]:
        sub_label = f'Threads: {num_threads}'
        time_results.append(benchmark.Timer(
            stmt='dr.dod_dl_forward(A, DOD_DL_model, DOD_DL_coeff_model, mu_0, nu_0, nt)',
            setup='from master_project_1 import dod_dl_rom as dr',
            globals={'A': A, 'DOD_DL_model': DOD_DL_model, 
                     'DOD_DL_coeff_model': DOD_DL_model, 'mu_0': mu_0, 'nu_0' : nu_0, 'nt': nt},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='Linear DOD DL ROM',
            ).blocked_autorange(min_run_time=0.5))
        time_results.append(benchmark.Timer(
            stmt='dr.pod_dl_forward(A, POD_DL_coeff_model, De_model, mu_0, nu_0, nt)',
            setup='from master_project_1 import dod_dl_rom as dr',
            globals={'A': A, 'POD_DL_coeff_model': POD_DL_coeff_model, 
                     'De_model': De_model, 'mu_0': mu_0, 'nu_0' : nu_0, 'nt': nt},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='POD DL ROM',
            ).blocked_autorange(min_run_time=0.5))
        time_results.append(benchmark.Timer(
            stmt='dr.colora_dl_forward(A, stat_DOD_model, stat_Coeff_model, CoLoRA_DL_model, mu_0, nu_0, nt)',
            setup='from master_project_1 import dod_dl_rom as dr',
            globals={'A': A, 'stat_DOD_model': stat_DOD_model, 'stat_Coeff_model': stat_Coeff_model, 
                     'CoLoRA_DL_model': CoLoRA_DL_model, 'mu_0': mu_0, 'nu_0' : nu_0, 'nt': nt},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='CoLoRA DL ROM',
            ).blocked_autorange(min_run_time=0.5))

    #region Get Solutions of all Models for Data Batch
    tqdm.write("Collecting Error...")
    coeff_dl_solutions = []
    pod_dl_solutions = []
    colora_dl_solutions = []
    norm_solutions = []
    proj_solutions = []
    solutions = []
    for entry in tqdm(performance_data, desc="Performance detection", leave=False):
        mu_i = torch.tensor(entry['mu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        mu_i = mu_i.unsqueeze(0)
        nu_i = torch.tensor(entry['nu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        nu_i = nu_i.unsqueeze(0)

        # True solution
        u_i = entry['solution']
        norm_u_i = l2_norm(u_i)
        proj_u_i = u_i @ G @ A_np @ A_np.T

        # Append Coeff_DL + POD_DL + CoLoRA_DL solution
        pod_dl_solutions.append(
            dr.pod_dl_forward(A, POD_DL_coeff_model, De_model, mu_i, nu_i, nt))
        colora_dl_solutions.append(
            dr.colora_dl_forward(A, stat_DOD_model, stat_Coeff_model, CoLoRA_DL_model, mu_i, nu_i, nt))
        coeff_dl_solutions.append(
            dr.dod_dl_forward(A, DOD_DL_model, DOD_DL_coeff_model, mu_i, nu_i, nt))
    
        # Append further relevant quantities
        norm_solutions.append(norm_u_i)
        proj_solutions.append(proj_u_i)
        solutions.append(u_i)
    
    # Append Error for Full sized Batch
    abs_diff_lin_dod_dl = [x - y for x, y in zip(solutions, coeff_dl_solutions)]
    abs_error_lin_dod_dl.append(error_loader(abs_diff_lin_dod_dl))
    rel_error_lin_dod_dl.append(error_loader([x / y for x, y in zip(abs_diff_lin_dod_dl, norm_solutions)]))

    abs_diff_pod_dl = [x - y for x, y in zip(solutions, pod_dl_solutions)]
    abs_error_pod_dl.append(error_loader(abs_diff_pod_dl))
    rel_error_pod_dl.append(error_loader([x / y for x, y in zip(abs_diff_pod_dl, norm_solutions)]))
    

    abs_diff_colora_dl = [x - y for x, y in zip(solutions, colora_dl_solutions)]
    abs_error_colora_dl.append(error_loader(abs_diff_colora_dl))
    rel_error_colora_dl.append(error_loader([x / y for x, y in zip(abs_diff_colora_dl, norm_solutions)]))
    
    ambient_diff = [x - y for x, y in zip(solutions, proj_solutions)]
    ambient_errors.append(error_loader([x / y for x, y in zip(ambient_diff, norm_solutions)]))
    '''=============potentially useless==============
    # Append Error for Ambient Batch
    proj_diff_lin_dod_dl = [x - y for x, y in zip(proj_solutions, coeff_dl_solutions)]
    ambient_abs_error_lin_dod_dl.append(error_loader(proj_diff_lin_dod_dl))
    ambient_rel_error_lin_dod_dl.append(error_loader([x / y for x, y in zip(proj_diff_lin_dod_dl, norm_solutions)]))
   
    prof_diff_colora_dl = [x - y for x, y in zip(proj_solutions, colora_dl_solutions)]
    ambient_abs_error_colora_dl.append(error_loader(prof_diff_colora_dl))
    ambient_rel_error_colora_dl.append(error_loader([x / y for x, y in zip(prof_diff_colora_dl, norm_solutions)]))
    '''

#endregion

'''
----------------
----------------
Plot Errors
----------------
----------------
'''
x = np.linspace(2, 8, 6, dtype=int)

if True:
    abs_error_pod_dl = [np.nan] * len(x)
    rel_error_pod_dl = [np.nan] * len(x)
    ambient_abs_error_pod_dl = [np.nan] * len(x)
    ambient_rel_error_pod_dl = [np.nan] * len(x)


# Define color and style map for consistency
plot_styles = {
    'POD DL':     {'color': 'blue',  'linestyle': '--'},
    'Linear DOD DL': {'color': 'green', 'linestyle': '-'},
    'CoLoRA DL':     {'color': 'red',   'linestyle': ':'},
    'Ambient Error': {'color': 'grey',  'linestyle': '-.'}
}

# Start 1x2 subplot grid (since we removed 2 plots)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs = axs.flatten()  # for easy indexing

# --- Plot 1: Absolute L2 Errors ---
axs[0].plot(x, abs_error_pod_dl, label='POD DL', **plot_styles['POD DL'])
axs[0].plot(x, abs_error_lin_dod_dl, label='Linear DOD DL', **plot_styles['Linear DOD DL'])
axs[0].plot(x, abs_error_colora_dl, label='CoLoRA DL', **plot_styles['CoLoRA DL'])
axs[0].set_title('Absolute $L^2$-Errors')
axs[0].set_xlabel('Reduced Dimension $n$')
axs[0].set_ylabel('Absolute Error')
axs[0].grid(True)
axs[0].legend()

# --- Plot 2: Relative L2 Errors ---
axs[1].plot(x, rel_error_pod_dl, label='POD DL', **plot_styles['POD DL'])
axs[1].plot(x, rel_error_lin_dod_dl, label='Linear DOD DL', **plot_styles['Linear DOD DL'])
axs[1].plot(x, rel_error_colora_dl, label='CoLoRA DL', **plot_styles['CoLoRA DL'])
axs[1].plot(x, ambient_errors, label='Ambient Error', **plot_styles['Ambient Error'])
axs[1].set_title('Relative $L^2$-Errors')
axs[1].set_xlabel('Reduced Dimension $n$')
axs[1].set_ylabel('Relative Error')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()

plt.savefig('/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/benchmarks/performance.png', dpi=300, bbox_inches='tight')

plt.show()

'''
--------------------
Cast Timing Results to File
--------------------
'''
with open("/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/benchmarks/benchmark_timed.pkl", "wb") as f:
    pickle.dump(time_results, f)
time_compare = benchmark.Compare(time_results)
time_compare.colorize()
time_compare.print()

