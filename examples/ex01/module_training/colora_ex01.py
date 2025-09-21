import torch
from master_project_1 import reduced_order_models as dr
import numpy as np

# Usage example
N_h = 5101
N_A = 64
N = 64
N_prime = 16
n = 4
Nt = 10
diameter = 0.02
parameter_mu_dim = 1
parameter_nu_dim = 1
# DOD+DFNN
preprocess_dim = 2
dod_structure = [32, 16]
df_layers = [16, 8]
# DOD-DL-ROM
dod_dl_df_layers = [32, 16, 8]
dod_in_channels = 1
dod_hidden_channels = 1
dod_lin_dim_ae = 0
dod_kernel = 3
dod_stride = 2
dod_padding = 1
# POD-DL-ROM
pod_df_layers = [32, 16, 8]
pod_in_channels = 1
pod_hidden_channels = 1
pod_lin_dim_ae = 0
pod_kernel = 3
pod_stride = 2
pod_padding = 1
# CoLoRA-ROM
L = 3
stat_m = 4
stat_dod_structure = [128, 64]
stat_phi_n_structure = [16, 8]

# Training Example
generalepochs = 50
generalrestarts = 10
generalpatience = 3

# Fetch Training and Validation set
train_valid_data = dr.FetchTrainAndValidSet(0.8, 'ex01', 'N_A_reduced')
stat_train_valid_data = dr.FetchTrainAndValidSet(0.8, 'ex01', 'reduced_stationary')

# Initialize and train the stationary DOD model
stat_DOD_model = dr.statDOD(parameter_mu_dim, preprocess_dim, N_prime, N_A, stat_dod_structure)
stat_DOD_Trainer = dr.statDODTrainer(stat_DOD_model, N_A, 
                                 stat_train_valid_data, 
                                 generalepochs, generalrestarts, learning_rate=1e-3, 
                                 batch_size=128, patience=generalpatience)
best_loss4 = stat_DOD_Trainer.train()

# Initialize and train the stationary Coefficient Finding model
stat_Coeff_model = dr.statHadamardNN(parameter_mu_dim, parameter_nu_dim, stat_m, 
                                     N_prime, stat_phi_n_structure)
stat_Coeff_Trainer = dr.statHadamardNNTrainer(stat_DOD_model, stat_Coeff_model, N_A,
                                        stat_train_valid_data,
                                        generalepochs, generalrestarts, learning_rate=1e-3, 
                                        batch_size=128, patience=generalpatience)
best_loss5 = stat_Coeff_Trainer.train()

# Initialize the CoLoRA_DL model
CoLoRA_DL_model = dr.CoLoRA(N_A, L, N_prime, parameter_nu_dim)

# Initialize the CoLoRA_DL trainer
CoLoRa_DL_Trainer = dr.CoLoRATrainer(stat_DOD_model, stat_Coeff_model, 
                                         CoLoRA_DL_model, train_valid_data,
                                         generalepochs, generalrestarts, learning_rate=1e-3, 
                                         batch_size=128, patience=generalpatience)

# Train the CoLoRA_DL
best_loss6 = CoLoRa_DL_Trainer.train()
print(f"Best validation loss: {best_loss6}")

# Save Modules
torch.save(stat_DOD_model.state_dict(), 'examples/ex01/state_dicts/stat_DOD_Module.pth')
torch.save(stat_Coeff_model.state_dict(), 'examples/ex01/state_dicts/stat_CoeffDOD_Module.pth')
torch.save(CoLoRA_DL_model.state_dict(), 'examples/ex01/state_dicts/CoLoRA_Module.pth')

