import torch
from master_project_1 import dod_dl_rom as dr
import numpy as np


# Usage example
N_h = 5101
N_A = 64
nt = 10
diameter = 0.02
n = 4
parameter_mu_dim = 1
parameter_nu_dim = 1
# linear DOD-DL-ROM
preprocess_dim = 2
lin_m = 4
lin_dod_structure = [64, 64, 64]
lin_dod_phi_n_structure = [16, 16, 8]
lin_phi_n_structure = [16, 8]
# POD-DL-ROM
pod_coeff_ae_structure = [32, 16, 8]
pod_in_channels = 1
pod_hidden_channels = 1
pod_lin_dim_ae = 0
pod_kernel = 3
pod_stride = 2
pod_padding = 1
# CoLoRA-DL-ROM
L = 3
stat_m = 4
stat_dod_structure = [128, 64]
stat_phi_n_structure = [16, 8]

# Training Example
generalepochs = 100
generalrestarts = 5
generalpatience = 2

# Fetch Training and Validation set
train_valid_data = dr.FetchReducedTrainAndValidSet(0.8, 'ex01')
stat_train_valid_data = dr.StatFetchReducedTrainAndValidSet(0.8, 'ex01')

# Initialize and train the stationary DOD model
stat_DOD_model = dr.DOD(parameter_mu_dim, preprocess_dim, n, N_A, stat_dod_structure)
stat_DOD_Trainer = dr.DODTrainer(stat_DOD_model, N_A, 
                                 stat_train_valid_data, 'ex01', 
                                 generalepochs, generalrestarts, learning_rate=1e-3, 
                                 batch_size=128, patience=generalpatience)
best_loss4 = stat_DOD_Trainer.train()

# Initialize and train the stationary Coefficient Finding model
stat_Coeff_model = dr.CoeffDOD(parameter_mu_dim, parameter_nu_dim, stat_m, n, stat_phi_n_structure)
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
print(f"Best validation loss: {best_loss6}")

# Save Modules
torch.save(stat_DOD_model.state_dict(), '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/state_dicts/stat_DOD_Module.pth')
torch.save(stat_Coeff_model.state_dict(), '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/state_dicts/stat_CoeffDOD_Module.pth')
torch.save(CoLoRA_DL_model.state_dict(), '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/state_dicts/CoLoRA_Module.pth')

