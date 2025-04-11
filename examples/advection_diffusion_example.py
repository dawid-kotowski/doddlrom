import torch
from master_project_1 import DOD_DL, DOD_DL_Trainer, Coeff_DOD_DL_Trainer, Coeff_DOD_DL, Encoder, Decoder, AE_DOD_DL_Trainer, FetchTrainAndValidSet
import numpy as np


# Usage example
N_h = 221
N_A = 64
rank = 10
L = 1
N = 16
n = 4
m = 4
parameter_mu_dim = 1
parameter_nu_dim = 1
preprocess_dim = 2
dod_structure = [128, 64]
phi_N_structure = [32, 16]
phi_n_structure = [16, 8]
nt = 10
diameter = 0.1

# Fetch Training and Validation set
train_valid_data = FetchTrainAndValidSet(0.8)

# Initialize the DOD model
DOD_DL_model = DOD_DL(1, parameter_mu_dim, dod_structure, N, N_A)

# Initialize the DOD trainer
DOD_DL_trainer = DOD_DL_Trainer(DOD_DL_model, train_valid_data, N_A, epochs=50, 
                                restart=10, learning_rate=1e-3, batch_size=128)

# Train the DOD model
best_loss = DOD_DL_trainer.train()
print(f"Best validation loss: {best_loss}")

# Initialize the Coefficient Finding model
Coeff_model = Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, N, phi_N_structure)

# Initialize the Coefficient Finding trainer
Coeff_trainer = Coeff_DOD_DL_Trainer(N_A, DOD_DL_model, Coeff_model,
                                   train_valid_data, epochs=50, restarts=10, 
                                   learning_rate=1e-3, batch_size=128)

# Train the Coefficient model
best_loss2 = Coeff_trainer.train()
print(f"Best validation loss: {best_loss2}")

# Initialize the AE Coefficient Finding model
En_model = Encoder(N, 1, 1, n, 1, kernel=3, stride=2, padding=1)
De_model = Decoder(N, 1, 1, n, 1, kernel=3, stride=2, padding=1)
AE_Coeff_model = Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, phi_n_structure)

# Initialize the AE Coefficient Finding trainer
AE_DOD_DL_trainer = AE_DOD_DL_Trainer(N_A, DOD_DL_model, AE_Coeff_model, En_model, De_model,
                                    train_valid_data, epochs=200, restarts=10, learning_rate=1e-3, batch_size=128)

# Train the AE Coefficient model
best_loss3 = AE_DOD_DL_trainer.train()
print(f"Best validation loss: {best_loss3}")

# Save the Models to a file
torch.save(DOD_DL_model.state_dict(), 'training/DOD_Module.pth')
torch.save(Coeff_model.state_dict(), 'training/DOD_Coefficient_Module.pth')
torch.save({
    'encoder': En_model.state_dict(),
    'decoder': De_model.state_dict(),
    'coeff_model': AE_Coeff_model.state_dict(),
}, 'training/AE_DOD_DL_Module.pth')


