import torch
from master_project_1 import reduced_order_models as dr
import numpy as np


# Usage example
N_h = 20201
N_A = 64
nt = 10
diameter = 0.01
n = 4
parameter_mu_dim = 3
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
generalepochs = 500
generalrestarts = 20
generalpatience = 4

# Fetch Training and Validation set
train_valid_data = dr.FetchReducedTrainAndValidSet(0.8, 'ex02')
stat_train_valid_data = dr.StatFetchReducedTrainAndValidSet(0.8, 'ex02')

# Initialize the POD DL ROM model
output = int(np.sqrt(N_A))
pod_num_layers = 0
while (output - int(np.sqrt(n)) > pod_lin_dim_ae):
    output = int(np.floor((output + 2*pod_padding - pod_kernel) / pod_stride) + 1)
    pod_num_layers += 1
En_model = dr.Encoder(N_A, pod_in_channels, pod_hidden_channels, n, pod_num_layers, pod_kernel, pod_stride, pod_padding)
De_model = dr.Decoder(N_A, pod_in_channels, pod_hidden_channels, n, pod_num_layers, pod_kernel, pod_stride, pod_padding)
POD_DL_coeff_model = dr.Coeff_AE(parameter_mu_dim, parameter_nu_dim, n, pod_coeff_ae_structure)

# Initialize the AE Coefficient Finding trainer
POD_DL_coeff_trainer = dr.POD_DL_Trainer(POD_DL_coeff_model, En_model, De_model,
                                    train_valid_data, 'ex02', 0.999,
                                    generalepochs, generalrestarts, learning_rate=1e-3, 
                                    batch_size=128, patience=generalpatience)

# Train the AE Coefficient model
best_loss3 = POD_DL_coeff_trainer.train()
print(f"Best validation loss: {best_loss3}")

# Save the Modules
torch.save({
    'encoder': En_model.state_dict(),
    'decoder': De_model.state_dict(),
    'coeff_model': POD_DL_coeff_model.state_dict(),
}, '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex02/state_dicts/POD_DL_Module.pth')