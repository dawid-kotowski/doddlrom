import torch
from master_project_1 import dod_dl_rom as dr
import numpy as np


# Usage example
N_h = 20201
N_A = 64
nt = 10
diameter = 0.01
L = 3
N = 16
n = 4
m = 4
parameter_mu_dim = 3
parameter_nu_dim = 1
preprocess_dim = 2
dod_structure = [64, 64]
phi_n_structure = [16, 8]
coeff_ae_structure = [32, 16, 8]
stat_dod_structure = [128, 64]
pod_in_channels = 1
pod_hidden_channels = 1
lin_dim_ae = 0
kernel = 3
stride = 2
padding = 1
# Training Example
generalepochs = 200
generalrestarts = 5
generalpatience = 2

# Fetch Training and Validation set
train_valid_data = dr.FetchReducedTrainAndValidSet(0.8, 'ex02')
stat_train_valid_data = dr.StatFetchReducedTrainAndValidSet(0.8, 'ex02')

# Initialize and train the stationary DOD model
stat_DOD_model = dr.DOD(parameter_mu_dim, preprocess_dim, n, N_A, stat_dod_structure)
stat_DOD_Trainer = dr.DODTrainer(stat_DOD_model, N_A, 
                                 stat_train_valid_data, 'ex02', 
                                 generalepochs, generalrestarts, learning_rate=1e-3, 
                                 batch_size=128, patience=generalpatience)
best_loss4 = stat_DOD_Trainer.train()

# Initialize and train the stationary Coefficient Finding model
stat_Coeff_model = dr.CoeffDOD(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)
stat_Coeff_Trainer = dr.CoeffDODTrainer(stat_DOD_model, stat_Coeff_model, N_A,
                                        stat_train_valid_data, 'ex02',
                                        generalepochs, generalrestarts, learning_rate=1e-3, 
                                        batch_size=128, patience=generalpatience)
best_loss5 = stat_Coeff_Trainer.train()

# Initialize the CoLoRA_DL model
CoLoRA_DL_model = dr.CoLoRA_DL(N_A, L, n, parameter_nu_dim)

# Initialize the CoLoRA_DL trainer
CoLoRa_DL_Trainer = dr.CoLoRA_DL_Trainer(N_A, stat_DOD_model, stat_Coeff_model, 
                                         CoLoRA_DL_model, train_valid_data, 'ex02',
                                         generalepochs, generalrestarts, learning_rate=1e-3, 
                                         batch_size=128, patience=generalpatience)

# Train the CoLoRA_DL
best_loss6 = CoLoRa_DL_Trainer.train()
print(f"Best validation loss: {best_loss6}")

# Save Modules
torch.save(stat_DOD_model.state_dict(), '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex02/state_dicts/stat_DOD_Module.pth')
torch.save(stat_Coeff_model.state_dict(), '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex02/state_dicts/stat_CoeffDOD_Module.pth')
torch.save(CoLoRA_DL_model.state_dict(), '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex02/state_dicts/CoLoRA_Module.pth')

