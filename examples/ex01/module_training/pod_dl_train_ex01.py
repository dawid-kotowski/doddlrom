import torch
from master_project_1 import dod_dl_rom as dr
import numpy as np


# Usage example
N_h = 5101
N_A = 64
nt = 10
diameter = 0.02
L = 3
N = 16
n = 4
m = 4
parameter_mu_dim = 1
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
generalepochs = 500
generalrestarts = 5
generalpatience = 5

# Fetch Training and Validation set
train_valid_data = dr.FetchReducedTrainAndValidSet(0.8, 'ex01')
stat_train_valid_data = dr.StatFetchReducedTrainAndValidSet(0.8, 'ex01')

# Initialize the POD DL ROM model
output = int(np.sqrt(N_A))
pod_num_layers = 0
while (output - int(np.sqrt(n)) > lin_dim_ae):
    output = int(np.floor((output + 2*padding - kernel) / stride) + 1)
    pod_num_layers += 1
En_model = dr.Encoder(N_A, pod_in_channels, pod_hidden_channels, n, pod_num_layers, kernel, stride, padding)
De_model = dr.Decoder(N_A, pod_in_channels, pod_hidden_channels, n, pod_num_layers, kernel, stride, padding)
POD_DL_coeff_model = dr.Coeff_AE(parameter_mu_dim, parameter_nu_dim, n, coeff_ae_structure)

# Initialize the AE Coefficient Finding trainer
POD_DL_coeff_trainer = dr.POD_DL_Trainer(POD_DL_coeff_model, En_model, De_model,
                                    train_valid_data, 'ex01', 0.999,
                                    generalepochs, generalrestarts, learning_rate=1e-2, 
                                    batch_size=128, patience=generalpatience)

# Train the AE Coefficient model
best_loss3 = POD_DL_coeff_trainer.train()
print(f"Best validation loss: {best_loss3}")

# Save the Modules
torch.save({
    'encoder': En_model.state_dict(),
    'decoder': De_model.state_dict(),
    'coeff_model': POD_DL_coeff_model.state_dict(),
}, '/home/sereom/Documents/University/Studies/Mathe/Wissenschaftliche Arbeiten/Master/Masterarbeit Ohlberger/Programming/master-project-1/examples/ex01/state_dicts/POD_DL_Module.pth')