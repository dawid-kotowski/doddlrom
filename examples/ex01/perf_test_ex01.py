import torch
from master_project_1 import dod_dl_rom as dr
import numpy as np
import matplotlib.pyplot as plt

# Fixed Constants
N_h = 365
N_A = 64
rank = 10
L = 3
dynamic_dim = 4
N = 16
# unspecified final dim n
m = 4
parameter_mu_dim = 1
parameter_nu_dim = 1
preprocess_dim = 2
dod_structure = [128, 64]
phi_N_structure = [32, 16]
phi_n_structure = [16, 8]
stat_dod_structure = [128, 64]
nt = 10
diameter = 0.1
generalepochs = 5
generalrestarts = 2

# Fetch Training and Validation set
train_valid_data = dr.FetchTrainAndValidSet(0.8, 'ex01')
stat_train_valid_data = dr.StatFetchTrainAndValidSet(0.8, 'ex01')

# Initialize random Performance Check set
loaded_data = np.load('examples/ex01/training_data/training_data_ex01.npy', allow_pickle=True)
np.random.shuffle(loaded_data) 
performance_data = loaded_data[:50]

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
abs_error_ae_dod_dl = []
rel_error_ae_dod_dl = []
abs_error_colora_dl = []
rel_error_colora_dl = []
ambient_abs_error_lin_dod_dl = []
ambient_rel_error_lin_dod_dl = []
ambient_abs_error_ae_dod_dl = []
ambient_rel_error_ae_dod_dl = []
ambient_abs_error_colora_dl = []
ambient_rel_error_colora_dl = []
for n in range(2, 8):
    # Initialize the DOD model
    DOD_DL_model = dr.DOD_DL(preprocess_dim, parameter_mu_dim, dod_structure, N, N_A)

    # Initialize the DOD trainer
    DOD_DL_trainer = dr.DOD_DL_Trainer(DOD_DL_model, train_valid_data, N_A, 'ex01',
                                    generalepochs, generalrestarts, learning_rate=1e-3, 
                                    batch_size=128)

    # Train the DOD model
    best_loss = DOD_DL_trainer.train()

    # Initialize the Coefficient Finding model
    Coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, N, phi_N_structure)

    # Initialize the Coefficient Finding trainer
    Coeff_trainer = dr.Coeff_DOD_DL_Trainer(N_A, DOD_DL_model, Coeff_model,
                                    train_valid_data, 'ex01', 
                                    generalepochs, generalrestarts, learning_rate=1e-3, 
                                    batch_size=128)

    # Train the Coefficient model
    best_loss2 = Coeff_trainer.train()

    # Initialize the AE Coefficient Finding model
    En_model = dr.Encoder(N, 1, 1, n, 1, kernel=3, stride=2, padding=1)
    De_model = dr.Decoder(N, 1, 1, n, 1, kernel=3, stride=2, padding=1)
    AE_Coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)

    # Initialize the AE Coefficient Finding trainer
    AE_DOD_DL_trainer = dr.AE_DOD_DL_Trainer(N_A, DOD_DL_model, AE_Coeff_model, En_model, De_model,
                                        train_valid_data, 'ex01',
                                        generalepochs, generalrestarts, learning_rate=1e-3, 
                                        batch_size=128)

    # Train the AE Coefficient model
    best_loss3 = AE_DOD_DL_trainer.train()

    # Initialize and train the stationary DOD model
    stat_DOD_model = dr.DOD(preprocess_dim, n, N_A, stat_dod_structure)
    stat_DOD_Trainer = dr.DODTrainer(stat_DOD_model, N_A, 
                                    stat_train_valid_data, 'ex01', 
                                    generalepochs, generalrestarts, learning_rate=1e-3, 
                                    batch_size=128)
    best_loss4 = stat_DOD_Trainer.train()

    # Initialize and train the stationary Coefficient Finding model
    stat_Coeff_model = dr.CoeffDOD(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)
    stat_Coeff_Trainer = dr.CoeffDODTrainer(stat_DOD_model, stat_Coeff_model, N_A,
                                            stat_train_valid_data, 'ex01',
                                            generalepochs, generalrestarts, learning_rate=1e-3, 
                                            batch_size=128)
    best_loss5 = stat_Coeff_Trainer.train()

    # Initialize the CoLoRA_DL model
    CoLoRA_DL_model = dr.CoLoRA_DL(N_A, L, dynamic_dim, parameter_nu_dim)

    # Initialize the CoLoRA_DL trainer
    CoLoRa_DL_Trainer = dr.CoLoRA_DL_Trainer(N_A, stat_DOD_model, stat_Coeff_model, 
                                            CoLoRA_DL_model, train_valid_data, 'ex01',
                                            generalepochs, generalrestarts, learning_rate=1e-3, 
                                            batch_size=128)

    # Train the CoLoRA_DL
    best_loss6 = CoLoRa_DL_Trainer.train()


    '''
    ---------------------
    Start of Evaluation on random set
    ---------------------
    '''

    # Set all to evalutate
    DOD_DL_model.eval()
    Coeff_model.eval()
    De_model.eval()
    AE_Coeff_model.eval()
    stat_DOD_model.eval()
    stat_Coeff_model.eval()
    CoLoRA_DL_model.eval()

    # Get Solutions
    coeff_dl_solutions = []
    ae_dl_solutions = []
    colora_dl_solutions = []
    G = np.load('examples/ex01/training_data/gram_matrix_ex01.npy', allow_pickle=True)
    A_np = np.load('examples/ex01/training_data/ambient_matrix_ex01.npy', allow_pickle=True)
    A = torch.tensor(A_np, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    for entry in performance_data:
        mu_i = torch.tensor(entry['mu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        mu_i = mu_i.unsqueeze(0)
        nu_i = torch.tensor(entry['nu'], dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        nu_i = nu_i.unsqueeze(0)

        # True solution
        u_i = entry['solution']
        norm_u_i = l2_norm(u_i)
        proj_u_i = u_i @ G @ A_np @ A_np.T

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
            decoded_output = De_model(coeff_n_output).squeeze(0)
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
        coeff_dl_solutions.append(coeff_dl_sol)

        ae_dl_sol = torch.stack(ae_dl_solution, dim=0)
        ae_dl_sol = ae_dl_sol.detach().numpy()
        ae_dl_solutions.append(ae_dl_sol)

        colora_dl_sol = torch.stack(colora_dl_solution, dim=0)
        colora_dl_sol = colora_dl_sol.detach().numpy()
        colora_dl_solutions.append(colora_dl_sol)
    
    # Append Error for Full sized Batch
    abs_error_lin_dod_dl.append(error_loader(u_i - coeff_dl_solutions))
    rel_error_lin_dod_dl.append(error_loader(u_i - coeff_dl_solutions) / norm_u_i)
    abs_error_ae_dod_dl.append(error_loader(u_i - ae_dl_solutions))
    rel_error_ae_dod_dl.append(error_loader(u_i - ae_dl_solutions) / norm_u_i)
    abs_error_colora_dl.append(error_loader(u_i - colora_dl_solutions))
    rel_error_colora_dl.append(error_loader(u_i - colora_dl_solutions) / norm_u_i)

    # Append Error for Ambient Batch
    ambient_abs_error_lin_dod_dl.append(error_loader(proj_u_i - coeff_dl_solutions))
    ambient_rel_error_lin_dod_dl.append(error_loader(proj_u_i - coeff_dl_solutions) / norm_u_i)
    ambient_abs_error_ae_dod_dl.append(error_loader(proj_u_i - ae_dl_solutions))
    ambient_rel_error_ae_dod_dl.append(error_loader(proj_u_i - ae_dl_solutions) / norm_u_i)
    ambient_abs_error_colora_dl.append(error_loader(proj_u_i - colora_dl_solutions))
    ambient_rel_error_colora_dl.append(error_loader(proj_u_i - colora_dl_solutions) / norm_u_i)

'''
----------------
----------------
Plot Errors
----------------
----------------
'''
x = np.linspace(2, 8, 6, dtype=int)

# Define color and style map for consistency
plot_styles = {
    'AE DOD DL':     {'color': 'blue',  'linestyle': '--'},
    'Linear DOD DL': {'color': 'green', 'linestyle': '-'},
    'CoLoRA DL':     {'color': 'red',   'linestyle': ':'},
}

# Start 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()  # for easy indexing

# --- Plot 1: Absolute L2 Errors ---
axs[0].plot(x, abs_error_ae_dod_dl, label='AE DOD DL', **plot_styles['AE DOD DL'])
axs[0].plot(x, abs_error_lin_dod_dl, label='Linear DOD DL', **plot_styles['Linear DOD DL'])
axs[0].plot(x, abs_error_colora_dl, label='CoLoRA DL', **plot_styles['CoLoRA DL'])
axs[0].set_title('Absolute $L^2$-Errors')
axs[0].set_xlabel('Reduced Dimension $n$')
axs[0].set_ylabel('Absolute Error')
axs[0].grid(True)
axs[0].legend()

# --- Plot 2: Relative L2 Errors ---
axs[1].plot(x, rel_error_ae_dod_dl, label='AE DOD DL', **plot_styles['AE DOD DL'])
axs[1].plot(x, rel_error_lin_dod_dl, label='Linear DOD DL', **plot_styles['Linear DOD DL'])
axs[1].plot(x, rel_error_colora_dl, label='CoLoRA DL', **plot_styles['CoLoRA DL'])
axs[1].set_title('Relative $L^2$-Errors')
axs[1].set_xlabel('Reduced Dimension $n$')
axs[1].set_ylabel('Relative Error')
axs[1].grid(True)
axs[1].legend()

# --- Plot 3: Absolute Ambient-Accounted Errors ---
axs[2].plot(x, ambient_abs_error_ae_dod_dl, label='AE DOD DL', **plot_styles['AE DOD DL'])
axs[2].plot(x, ambient_abs_error_lin_dod_dl, label='Linear DOD DL', **plot_styles['Linear DOD DL'])
axs[2].plot(x, ambient_abs_error_colora_dl, label='CoLoRA DL', **plot_styles['CoLoRA DL'])
axs[2].set_title('Ambient Absolute $L^2$-Errors')
axs[2].set_xlabel('Reduced Dimension $n$')
axs[2].set_ylabel('Absolute Error')
axs[2].grid(True)
axs[2].legend()

# --- Plot 4: Relative Ambient-Accounted Errors ---
axs[3].plot(x, ambient_rel_error_ae_dod_dl, label='AE DOD DL', **plot_styles['AE DOD DL'])
axs[3].plot(x, ambient_rel_error_lin_dod_dl, label='Linear DOD DL', **plot_styles['Linear DOD DL'])
axs[3].plot(x, ambient_rel_error_colora_dl, label='CoLoRA DL', **plot_styles['CoLoRA DL'])
axs[3].set_title('Ambient Relative $L^2$-Errors')
axs[3].set_xlabel('Reduced Dimension $n$')
axs[3].set_ylabel('Relative Error')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()
    





