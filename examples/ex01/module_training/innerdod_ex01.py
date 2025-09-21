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

# Initialize the DOD model
DOD_model = dr.innerDOD(preprocess_dim, parameter_mu_dim, dod_structure, N_prime, N_A)

# Initialize the DOD trainer
DOD_trainer = dr.innerDODTrainer(DOD_model, train_valid_data,
                                   generalepochs, generalrestarts, learning_rate=1e-4, 
                                   batch_size=128, patience=generalpatience)

# Train the DOD model
best_loss = DOD_trainer.train()
print(f"Best validation loss: {best_loss}")

# Save the Models to a file
torch.save(DOD_model.state_dict(), 'examples/ex01/state_dicts/DOD_Module.pth')