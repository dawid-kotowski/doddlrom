import torch
from master_project_1 import dod_dl_rom as dr
import numpy as np


# Usage example
N_h = 365
N_A = 64
rank = 10
L = 3
dynamic_dim = 4
N = 16
n = 4
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

# Fetch Training and Validation set
train_valid_data = dr.FetchReducedTrainAndValidSet(0.8, 'ex01')
stat_train_valid_data = dr.StatFetchTrainAndValidSet(0.8, 'ex01')

# Initialize the DOD model
DOD_DL_model = dr.DOD_DL(preprocess_dim, parameter_mu_dim, dod_structure, N, N_A)

# Initialize the DOD trainer
DOD_DL_trainer = dr.DOD_DL_Trainer(DOD_DL_model, train_valid_data, N_A, 'ex01',
                                   epochs=50,restart=3, learning_rate=1e-3, 
                                   batch_size=128)

# Train the DOD model
best_loss = DOD_DL_trainer.train()
print(f"Best validation loss: {best_loss}")

# Initialize the Coefficient Finding model
Coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, N, phi_N_structure)

# Initialize the Coefficient Finding trainer
Coeff_trainer = dr.Coeff_DOD_DL_Trainer(N_A, DOD_DL_model, Coeff_model,
                                   train_valid_data, 'ex01', 
                                   epochs=50, restarts=3, learning_rate=1e-3, 
                                   batch_size=128)

# Train the Coefficient model
best_loss2 = Coeff_trainer.train()
print(f"Best validation loss: {best_loss2}")

# Initialize the AE Coefficient Finding model
En_model = dr.Encoder(N, 1, 1, n, 1, kernel=3, stride=2, padding=1)
De_model = dr.Decoder(N, 1, 1, n, 1, kernel=3, stride=2, padding=1)
AE_Coeff_model = dr.Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)

# Initialize the AE Coefficient Finding trainer
AE_DOD_DL_trainer = dr.AE_DOD_DL_Trainer(N_A, DOD_DL_model, AE_Coeff_model, En_model, De_model,
                                    train_valid_data, 'ex01',
                                    epochs=50, restarts=3, learning_rate=1e-3, 
                                    batch_size=128)

# Train the AE Coefficient model
best_loss3 = AE_DOD_DL_trainer.train()
print(f"Best validation loss: {best_loss3}")

# Initialize and train the stationary DOD model
stat_DOD_model = dr.DOD(preprocess_dim, n, N_A, stat_dod_structure)
stat_DOD_Trainer = dr.DODTrainer(stat_DOD_model, N_A, 
                                 stat_train_valid_data, 'ex01', 
                                 epochs=50, restart=3, learning_rate=1e-3, 
                                 batch_size=128)
best_loss4 = stat_DOD_Trainer.train()

# Initialize and train the stationary Coefficient Finding model
stat_Coeff_model = dr.CoeffDOD(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)
stat_Coeff_Trainer = dr.CoeffDODTrainer(stat_DOD_model, stat_Coeff_model, N_A,
                                        stat_train_valid_data, 'ex01',
                                        epochs=50, restarts=3, learning_rate=1e-3, 
                                        batch_size=128)
best_loss5 = stat_Coeff_Trainer.train()

# Initialize the CoLoRA_DL model
CoLoRA_DL_model = dr.CoLoRA_DL(N_A, L, dynamic_dim, parameter_nu_dim)

# Initialize the CoLoRA_DL trainer
CoLoRa_DL_Trainer = dr.CoLoRA_DL_Trainer(N_A, stat_DOD_model, stat_Coeff_model, 
                                         CoLoRA_DL_model, train_valid_data, 'ex01',
                                         epochs=50, restarts=3, learning_rate=1e-3, 
                                         batch_size=128)

# Train the CoLoRA_DL
best_loss6 = CoLoRa_DL_Trainer.train()
print(f"Best validation loss: {best_loss6}")

# Save the Models to a file
torch.save(DOD_DL_model.state_dict(), 'examples/ex01/state_dicts/DOD_Module.pth')
torch.save(Coeff_model.state_dict(), 'examples/ex01/state_dicts/DOD_Coefficient_Module.pth')
torch.save({
    'encoder': En_model.state_dict(),
    'decoder': De_model.state_dict(),
    'coeff_model': AE_Coeff_model.state_dict(),
}, 'examples/ex01/state_dicts/AE_DOD_DL_Module.pth')
torch.save(stat_DOD_model.state_dict(), 'examples/ex01/state_dicts/stat_DOD_Module.pth')
torch.save(stat_Coeff_model.state_dict(), 'examples/ex01/state_dicts/stat_CoeffDOD_Module.pth')
torch.save(CoLoRA_DL_model.state_dict(), 'examples/ex01/state_dicts/CoLoRA_Module.pth')


