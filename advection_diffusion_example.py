import torch
from dod_dl_rom import DOD_DL_Trainer, DOD_DL, Coeff_DOD_DL_Trainer, Coeff_DOD_DL
import numpy as np


class FetchTrainAndValidSet:
    def __init__(self, train_to_val_ratio):
        # 0.8 = train_to_val_ratio means 80 % of training data and 20 % of validation data
        self.train_to_val_ratio = train_to_val_ratio
        loaded_data = np.load('training_data.npy', allow_pickle=True)
        np.random.shuffle(loaded_data)
        num_samples = len(loaded_data)
        num_train_samples = int(train_to_val_ratio * num_samples)
        self.training_data = loaded_data[:num_train_samples]
        self.validation_data = loaded_data[num_train_samples:]


    def __call__(self, set_type):
        if set_type == 'train':
            return self.training_data
        elif set_type == 'valid':
            return self.validation_data
        else:
            return 'Type Undefined'


# Usage example
N_h = 41
rank = 5
L = 2
n = 4
m = 5
parameter_mu_dim = 1
parameter_nu_dim = 1
preprocess_dim = 41

# Fetch Training and Validation set
train_valid_data = FetchTrainAndValidSet(0.8)

# Initialize the DOD model
DOD_DL_model = DOD_DL(N_h, L, preprocess_dim, parameter_mu_dim, rank, parameter_nu_dim, n)

# Initialize the DOD trainer
DOD_DL_trainer = DOD_DL_Trainer(DOD_DL_model, train_valid_data, 50, 5, 1e-3, 32)

# Train the DOD model
best_loss = DOD_DL_trainer.train()
print(f"Best validation loss: {best_loss}")

# Initialize the Coefficient Finding model
Coeff_model = Coeff_DOD_DL(parameter_mu_dim, parameter_nu_dim, m, n, [20])

# Initialize the Coefficient Finding trainer
Coeff_trainer = Coeff_DOD_DL_Trainer(DOD_DL_model, Coeff_model,
                                   train_valid_data, 50, 5, 1e-3, 32)

# Train the Coefficient model
best_loss2 = Coeff_trainer.train()
print(f"Best validation loss: {best_loss2}")

# Save the Models to a file
torch.save(DOD_DL_model.state_dict(), 'DOD_Module.pth')
torch.save(Coeff_model.state_dict(), 'DOD_Coefficient_Module.pth')


