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
generalepochs = 200
generalrestarts = 5
generalpatience = 2

# Fetch Training and Validation set
train_valid_data = dr.FetchReducedTrainAndValidSet(0.8, 'ex01')
stat_train_valid_data = dr.StatFetchReducedTrainAndValidSet(0.8, 'ex01')

# Initialize the DOD model
DOD_DL_model = dr.DOD_DL(preprocess_dim, parameter_mu_dim, dod_structure, n, N_A)

# Initialize the DOD trainer
DOD_DL_trainer = dr.DOD_DL_Trainer(DOD_DL_model, train_valid_data, N_A, 'ex01',
                                   generalepochs,generalrestarts, learning_rate=1e-4, 
                                   batch_size=128, patience=generalpatience)

# Train the DOD model
best_loss = DOD_DL_trainer.train()
print(f"Best validation loss: {best_loss}")

# Initialize the Coefficient Finding model
DOD_DL_coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)

# Initialize the Coefficient Finding trainer
DOD_DL_coeff_trainer = dr.Coeff_DOD_DL_Trainer(N_A, DOD_DL_model, DOD_DL_coeff_model,
                                   train_valid_data, 'ex01', 
                                   generalepochs, generalrestarts, learning_rate=1e-3, 
                                   batch_size=128, patience=generalpatience)

# Train the Coefficient model
best_loss2 = DOD_DL_coeff_trainer.train()
print(f"Best validation loss: {best_loss2}")

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
print(f"Best validation loss: {best_loss6}")

# Save the Models to a file
torch.save(DOD_DL_model.state_dict(), 'examples/ex01/state_dicts/DOD_Module.pth')
torch.save(DOD_DL_coeff_model.state_dict(), 'examples/ex01/state_dicts/DOD_Coefficient_Module.pth')
torch.save({
    'encoder': En_model.state_dict(),
    'decoder': De_model.state_dict(),
    'coeff_model': POD_DL_coeff_model.state_dict(),
}, 'examples/ex01/state_dicts/POD_DL_Module.pth')
torch.save(stat_DOD_model.state_dict(), 'examples/ex01/state_dicts/stat_DOD_Module.pth')
torch.save(stat_Coeff_model.state_dict(), 'examples/ex01/state_dicts/stat_CoeffDOD_Module.pth')
torch.save(CoLoRA_DL_model.state_dict(), 'examples/ex01/state_dicts/CoLoRA_Module.pth')


