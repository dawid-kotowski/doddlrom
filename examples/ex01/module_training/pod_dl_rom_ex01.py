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
train_valid_data = dr.FetchTrainAndValidSet(0.8, 'ex01', 'N_reduced')

# Initialize the POD DL ROM model
output = int(np.sqrt(N))
pod_num_layers = 0
while (output - int(np.sqrt(n)) > pod_lin_dim_ae):
    output = int(np.floor((output + 2*pod_padding - pod_kernel) / pod_stride) + 1)
    pod_num_layers += 1
En_model = dr.Encoder(N, pod_in_channels, pod_hidden_channels, n, pod_num_layers, pod_kernel, pod_stride, pod_padding)
De_model = dr.Decoder(N, pod_in_channels, pod_hidden_channels, n, pod_num_layers, pod_kernel, pod_stride, pod_padding)
DFNN_P_n_model = dr.DFNN(parameter_mu_dim, parameter_nu_dim, n, pod_df_layers)

# Initialize the AE Coefficient Finding trainer
DFNN_P_n_trainer = dr.POD_DL_ROMTrainer(DFNN_P_n_model, En_model, De_model,
                                    train_valid_data, 0.999,
                                    generalepochs, generalrestarts, learning_rate=1e-2, 
                                    batch_size=128, patience=generalpatience)

# Train the AE Coefficient model
best_loss3 = DFNN_P_n_trainer.train()
print(f"Best validation loss: {best_loss3}")

# Save the Modules
torch.save({
    'encoder': En_model.state_dict(),
    'decoder': De_model.state_dict(),
    'coeff_model': DFNN_P_n_model.state_dict(),
}, 'examples/ex01/state_dicts/POD_DL_ROM_Module.pth')