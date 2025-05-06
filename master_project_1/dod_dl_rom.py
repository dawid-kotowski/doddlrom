import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
import math
import copy
from torch.utils.data import Dataset, DataLoader

# Some Constants
time_end = 1.
nt = 10

# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Define Training/Validation Splitter
class FetchReducedTrainAndValidSet:
    def __init__(self, train_to_val_ratio, example_name):
        # 0.8 = train_to_val_ratio means 80 % of training data and 20 % of validation data
        self.train_to_val_ratio = train_to_val_ratio
        path = f'examples/{example_name}/training_data/reduced_training_data_{example_name}.npy'
        loaded_data = np.load(path, allow_pickle=True)
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

# Define Training/Validation Splitter for the stationary model
class StatFetchTrainAndValidSet:
    def __init__(self, train_to_val_ratio, example_name):
        # 0.8 = train_to_val_ratio means 80 % of training data and 20 % of validation data
        self.train_to_val_ratio = train_to_val_ratio
        path = f'examples/{example_name}/training_data/stationary_training_data_{example_name}.npy'
        loaded_data = np.load(path, allow_pickle=True)
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

# Define DatasetLoader
class DatasetLoader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        mu = torch.tensor(entry['mu'], dtype=torch.float32)
        nu = torch.tensor(entry['nu'], dtype=torch.float32)
        solution = torch.tensor(entry['solution'], dtype=torch.float32)
        return mu, nu, solution

# Define DatasetLoader for stationary DOD
class StatReducedDatasetLoader(Dataset):
    def __init__(self, data, G, A, N_A):
        self.data = data
        self.G = G
        self.A = A
        self.N_A = N_A

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        mu = torch.tensor(entry['mu'], dtype=torch.float32)
        u = torch.tensor(entry['solution'], dtype=torch.float32)
        u_new = torch.matmul(self.A.T, torch.matmul(self.G, u))
        u_new = u_new[:self.N_A]
        return mu, u_new

''' 
------------------------------------
In the following a DOD_DL is introduced, which dynamically approximates a reduced basis
w.r.t the geometric parameter mu

V: (Theta \times Gamma) -> (R^(N_h \times N))
------------------------------------
'''

# Define Seed Module for geometric parameter
class SeedModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SeedModule, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, mu):
        mu = func.leaky_relu(self.fc(mu), 0.1)
        return mu

# Define Root Modules for DOD_DL
class RootModule(nn.Module):
    def __init__(self, seed_dim, root_dim, root_layer_sizes, leaky_relu_slope=0.1):
        super(RootModule, self).__init__()
        if root_layer_sizes is None:
            root_layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(seed_dim + 1, root_layer_sizes[0]))
        for i in range(len(root_layer_sizes) - 1):
            self.layers.append(nn.Linear(root_layer_sizes[i], root_layer_sizes[i + 1]))
            if i < len(root_layer_sizes) - 1:
                self.layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        self.layers.append(nn.Linear(root_layer_sizes[len(root_layer_sizes) - 1], root_dim))

    def forward(self, mu_t):
        for layer_forward in self.layers:
            mu_t = layer_forward(mu_t)
        return mu_t

# Define Complete DOD_DL_ DL Model 
# returns a tensor of size [N_A, N]
class DOD_DL(nn.Module):
    def __init__(self, seed_dim, geometric_dim, root_layer_sizes, N, N_A):
        super(DOD_DL, self).__init__()
        self.seed_module = SeedModule(geometric_dim, seed_dim)
        self.root_modules = nn.ModuleList([RootModule(seed_dim, N_A, root_layer_sizes) for _ in range(N)])

    def forward(self, mu, t):
        seed_output = self.seed_module(mu)
        mu_t = torch.cat((seed_output, t), dim=1)
        root_outputs = [root(mu_t) for root in self.root_modules]
        v_mu = torch.stack(root_outputs, dim=0).transpose(0,1)
        return v_mu

# Define DOD_DL_-DL training
class DOD_DL_Trainer:
    def __init__(self, nn_model, train_valid_set, N_A, example_name, epochs=1, restart=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.restarts = restart
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = nn_model.to(device)
        self.device = device

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, solution_batch):
        batch_size = mu_batch.size(0)  # Get the batch size (can be problem, if set is non-divisible)
        temp_loss = 0.

        for i in range(nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i * time_end/(nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            output = self.model(mu_batch, t_batch)

            # Perform batch matrix multiplications
            v_u = torch.bmm(output, solution_batch[:, i, :].unsqueeze(2))
            u_proj = torch.bmm(output.transpose(1, 2), v_u)

            error = solution_batch[:, i, :].unsqueeze(2) - u_proj
            temp_loss += torch.sum(torch.norm(error, dim=1) ** 2)

        loss = temp_loss / (batch_size * (nt + 1))

        return loss

    def train(self):
        best_model = None
        best_loss = float('inf')

        for _ in range(self.restarts):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Model: DOD_DL, Restart: {_ + 1}, Epoch: {epoch} Loss: {total_loss / len(self.train_loader)}")

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for mu_batch, nu_batch, solution_batch in self.valid_loader:
                    val_loss += self.loss_function(mu_batch, solution_batch).item()
                val_loss /= len(self.valid_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()

            print(f'Restart DOD_DL. Gen Count at {_ + 1} with current best loss {best_loss}')

        # Load the best model
        self.model.load_state_dict(best_model)
        return best_loss

''' 
------------------------------------
Adding to the DOD_DL is a network trying to approximate the latent dynamics of the underlying 
n-dim solution manifold. We present the following different approaches for this

Coeff_DL     : (Theta \times Theta' \times \Gamma) -> R^N
             ; Linear(Linear(mu_t)) @ Linear(Linear(nu_t)) \mapsto u_N

AE_DL        : (Theta \times Theta' \times \Gamma) -> R^N
             ; Encoder(Linear(Linear(mu_t)) @ Linear(Linear(nu_t))) \mapsto u_N

AE_CoLoRA_DL : (Theta \times Theta' \times \Gamma) -> R^N
             ; Encoder(CoLoRA(mu, nu, t)) \mapsto u_N
------------------------------------
'''

# Define Phi_1 module for allocation
class Phi1Module(nn.Module):
    def __init__(self, geometric_dim, m_0, n_0, layer_sizes, leaky_relu_slope=0.1):
        super(Phi1Module, self).__init__()
        self.m = m_0
        self.n = n_0
        if layer_sizes is None:
            layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(geometric_dim + 1, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 1:
                self.layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        self.layers.append(nn.Linear(layer_sizes[len(layer_sizes) - 1], n_0 * m_0))

    def forward(self, mu_t):
        for layer_forward in self.layers:
            mu_t = layer_forward(mu_t)
        mu_t = mu_t.view(-1, self.m, self.n)
        return mu_t

# Define Phi_2 module for allocation
class Phi2Module(nn.Module):
    def __init__(self, physical_dim, m_0, n_0, layer_sizes, leaky_relu_slope=0.1):
        super(Phi2Module, self).__init__()
        self.m = m_0
        self.n = n_0
        if layer_sizes is None:
            layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(physical_dim + 1, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 1:
                self.layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        self.layers.append(nn.Linear(layer_sizes[len(layer_sizes) - 1], n_0 * m_0))

    def forward(self, nu_t):
        for layer_forward in self.layers:
            nu_t = layer_forward(nu_t)
        nu_t = nu_t.view(-1, self.m, self.n)
        return nu_t

# Define Complete parameter-to-DOD_DL-coefficients Model 
# returns [B, n]
class Coeff_DOD_DL(nn.Module):
    def __init__(self, geometric_dim, physical_dim, m_0, n_0, layer_sizes=None):
        super(Coeff_DOD_DL, self).__init__()
        self.phi_1_module = Phi1Module(geometric_dim, m_0, n_0, layer_sizes)
        self.phi_2_module = Phi2Module(physical_dim, m_0, n_0, layer_sizes)

    def forward(self, mu, nu, t):
        mu_t = torch.cat((mu, t), dim=1)
        nu_t = torch.cat((nu, t), dim=1)
        phi_1 = self.phi_1_module(mu_t)
        phi_2 = self.phi_2_module(nu_t)
        phi = phi_1 * phi_2
        phi_sum = torch.sum(phi, dim=1).squeeze()
        return phi_sum

# Define the Trainer
class Coeff_DOD_DL_Trainer:
    def __init__(self, N_A, DOD_DL_model, coeffnn_model, train_valid_set, example_name, epochs, restarts, learning_rate,
                 batch_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.DOD_DL = DOD_DL_model.to(device)
        self.model = coeffnn_model.to(device)
        self.device = device

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)
        temp_error = 0.0
        for i in range(nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i * time_end / (nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            # Get the coefficient model output; expected shape: (B, n)
            output = self.model(mu_batch, nu_batch, t_batch)
            # Get the DOD_DL_ model output; expected shape: (B, n, N_A)
            DOD_DL_output = self.DOD_DL(mu_batch, t_batch)

            # Extract the solution slice at time step i; expected shape: (B, N_A)
            u_ambient_proj = solution_batch[:, i, :].unsqueeze(2)

            # u_proj: (B, n, 1) then squeezed to (B, n)
            u_proj = (torch.bmm(DOD_DL_output, u_ambient_proj)).squeeze(2)

            error = output - u_proj  # (B, n)
            temp_error += torch.sum(torch.norm(error, dim=1) ** 2)

        loss = temp_error / batch_size
        return loss

    def train(self):
        best_model = None
        best_loss = float('inf')

        for _ in range(self.restarts):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Model: Coeff_DL, Restart: {_ + 1}, Epoch: {epoch} Loss: {total_loss / len(self.train_loader)}")

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for mu_batch, nu_batch, solution_batch in self.valid_loader:
                    val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                val_loss /= len(self.valid_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()

            print(f'Restart Coefficient Finder. Gen Count at {_ + 1} with current best loss {best_loss}')

        # Load the best model
        self.model.load_state_dict(best_model)
        return best_loss

# Define Encoder 
# takes [B, input_dim] return [B, loop(floor((input_dim + 2p - k) / s) + 1)] 
class Encoder(nn.Module):
    def __init__(self, input_dim, in_channels, hidden_channels, latent_dim,
                num_layers, kernel, stride, padding):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.padding = padding
        self.kernel = kernel
        self.stride = stride
        self.grid_size = self._compute_grid(input_dim)
        H, W = self.grid_size

        # Convolutional encoder
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            next_channels = hidden_channels
            layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=kernel,
                                    stride=stride, padding=padding))
            layers.append(nn.LeakyReLU(0.1))
            current_channels = next_channels

        self.encoder = nn.Sequential(*layers)

        # Compute output shape after all conv layers
        C_prime, H_prime, W_prime = self._compute_encoder_shape()
        
        self.linear = nn.Linear(C_prime * H_prime * W_prime, latent_dim)

    def _compute_grid(self, D):
        size = math.ceil(math.sqrt(D))
        return (size, size)
    
    def _compute_encoder_shape(self):
        D_temp = math.ceil(math.sqrt(self.input_dim))
        for layer in range(self.num_layers):
            D_temp = math.floor((D_temp + 2 * self.padding - self.kernel) / self.stride) + 1
        return (self.hidden_channels, D_temp, D_temp)

    def forward(self, x1d):
        B, D = x1d.shape
        total_grid = self.grid_size[0] * self.grid_size[1]
        if D < total_grid:
            x1d = func.pad(x1d, (0, total_grid - D))
        elif D > total_grid:
            raise ValueError("Input dimension exceeds grid capacity.")
        
        x2d = x1d.view(B, 1, *self.grid_size)
        conv_out = self.encoder(x2d).view(B, -1)
        z = self.linear(conv_out)  # Latent vector
        return z

# Define Decoder 
# takes [B, loop(floor((input_dim + 2p - k) / s) + 1)] return [B, input_dim]
class Decoder(nn.Module):
    def __init__(self, output_dim, out_channels, hidden_channels, latent_dim, 
                 num_layers, kernel, stride, padding):
        super().__init__()
        self.output_dim = output_dim
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.padding = padding
        self.kernel = kernel
        self.stride = stride
        self.encoder_output_shape = self._compute_encoder_shape()  # (C, H, W)
        C, H, W = self.encoder_output_shape
        self.linear = nn.Linear(latent_dim, C * H * W)

        layers = []
        current_channels = C
        for i in range(num_layers):
            next_channels = hidden_channels if i < num_layers - 1 else out_channels
            layers.append(nn.ConvTranspose2d(current_channels, next_channels, 
                                             kernel_size=kernel, stride=stride, padding=padding, output_padding=1))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU(0.1))
            else:
                layers.append(nn.Sigmoid())
            current_channels = next_channels

        self.decoder = nn.Sequential(*layers)

    def _compute_encoder_shape(self):
        D_temp = math.ceil(math.sqrt(self.output_dim))
        for layer in range(self.num_layers):
            D_temp = math.floor((D_temp + 2 * self.padding - self.kernel) / self.stride) + 1
        return (self.out_channels, D_temp, D_temp)

    def forward(self, z):
        B = z.shape[0]
        x_unflat = self.linear(z)
        x_unflat = x_unflat.view(B, *self.encoder_output_shape)
        x_recon = self.decoder(x_unflat)
        return x_recon.view(B, -1)[:, :self.output_dim]
    
# Define the Trainer
class AE_DOD_DL_Trainer:
    def __init__(self, N_A, DOD_DL_model, Coeff_DOD_DL_model, Encoder_model, Decoder_model, train_valid_set, example_name, error_weight=0.5,
                 epochs=1, restarts=1, learning_rate=1e-3, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.error_weight = error_weight
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.DOD_DL = DOD_DL_model.to(device)
        self.en_model = Encoder_model.to(device)
        self.de_model = Decoder_model.to(device)
        self.coeff_model = Coeff_DOD_DL_model.to(device)
        self.device = device

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)
        temp_error = 0.0
        for i in range(nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i * time_end / (nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            # Get the coefficient model output; expected shape: (B, n)
            coeff_output = self.coeff_model(mu_batch, nu_batch, t_batch)
            # Get the non linear expansion output; expected shape: (B, N)
            decoder_output = self.de_model(coeff_output)
            # Get the DOD_DL_ model output; expected shape: (B, N, N_A)
            DOD_DL_output = self.DOD_DL(mu_batch, t_batch)

            # Extract the solution slice at time step i; expected shape: (B, N_A)
            solution_slice = solution_batch[:, i, :]
            u_ambient_proj = solution_slice.unsqueeze(2) # shape: (B, N_A, 1)

            # Get the Dynmaics output; expected shape: (B, N)
            dynamics_proj = torch.bmm(DOD_DL_output, u_ambient_proj).squeeze(2)
            # Encoder output; expected shape: (B, N)
            encoder_output = self.en_model(dynamics_proj)

            dynam_error = dynamics_proj - decoder_output  # (B, n)
            proj_error = encoder_output - coeff_output # (B, n)
            temp_error += (self.error_weight / 2 * torch.sum(torch.norm(dynam_error, dim=1) ** 2) 
                           + (1-self.error_weight) / 2 * torch.sum(torch.norm(proj_error, dim=1) ** 2))

        loss = temp_error / batch_size
        return loss

    def train(self):
        best_model = None
        best_loss = float('inf')

        # Multiple restarts loop
        for restart in range(self.restarts):
            # Reinitialize the trainable models (except DOD_DL_model which remains fixed)
            self.coeff_model.apply(initialize_weights)
            self.en_model.apply(initialize_weights)
            self.de_model.apply(initialize_weights)
            
            # Combine parameters of the three modules for the optimizer
            params = list(self.coeff_model.parameters()) + \
                     list(self.en_model.parameters()) + \
                     list(self.de_model.parameters())
            optimizer = optim.Adam(params, lr=self.learning_rate)
            
            for epoch in range(self.epochs):
                # Set all models to train mode
                self.coeff_model.train()
                self.en_model.train()
                self.de_model.train()
                self.DOD_DL.train()  # Even if not trained, set to train for consistency
                
                total_loss = 0.0
                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    # Transfer data to device
                    mu_batch = mu_batch.to(self.device)
                    nu_batch = nu_batch.to(self.device)
                    solution_batch = solution_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(self.train_loader)
                print(f"Model: AE_DOD_DL, Restart: {restart + 1}, Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")
                
            # Evaluate on validation set
            self.coeff_model.eval()
            self.en_model.eval()
            self.de_model.eval()
            self.DOD_DL.eval()
            
            val_loss = 0.0
            with torch.no_grad():
                for mu_batch, nu_batch, solution_batch in self.valid_loader:
                    mu_batch = mu_batch.to(self.device)
                    nu_batch = nu_batch.to(self.device)
                    solution_batch = solution_batch.to(self.device)
                    val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                avg_val_loss = val_loss / len(self.valid_loader)
                print(f"Restart: {restart + 1}, Validation Loss: {avg_val_loss:.4f}")
            
            # Save the best model (only saving coeff, encoder, and decoder)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = {
                    "encoder": copy.deepcopy(self.en_model.state_dict()),
                    "decoder": copy.deepcopy(self.de_model.state_dict()),
                    "coeff_model": copy.deepcopy(self.coeff_model.state_dict())
                }
                print(f"Updated best AE_DOD_DL model at restart {restart + 1} with validation loss {best_loss:.4f}")
        
        # Load the best model's state dictionaries into the models
        if best_model is not None:
            self.en_model.load_state_dict(best_model["encoder"])
            self.de_model.load_state_dict(best_model["decoder"])
            self.coeff_model.load_state_dict(best_model["coeff_model"])
            print("Loaded best model state dicts.")
        else:
            print("No best model was found.")
            
        return best_loss

'''
-----------------------------------
Having considered the DOD-algorithm in analogy to the POD-DL-ROM 
one might instead go with a rather differently influenced idea;
assuming we have sufficent quick KnW decay regarding the time-separated snapshot
manifold, one can instead of relying on the direct Input of a NN to 
introduce time dependency, try achieving this using a CoLoRA inspired 
architecture with a previous stationary solution starting point, i.e.

u(mu,  nu, t) = C_L(...C_2(C_1(V_0(mu) phi_0(mu, nu))))

with C_i(X) := W_i X + A_i diag(a_i(nu, t)) B_i X + b_i     for i <= L
-----------------------------------
'''
# Define the stationary DOD for the stationary equivalent of the problem
class StatSeedModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StatSeedModule, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = func.leaky_relu(self.fc(x), 0.1)
        return x
class StatRootModule(nn.Module):
    def __init__(self, input_dim, output_dim, root_layer_sizes, leaky_relu_slope=0.1):
        super(StatRootModule, self).__init__()
        if root_layer_sizes is None:
            root_layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, root_layer_sizes[0]))
        for i in range(len(root_layer_sizes) - 1):
            self.layers.append(nn.Linear(root_layer_sizes[i], root_layer_sizes[i + 1]))
            if i < len(root_layer_sizes) - 1:
                self.layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        self.layers.append(nn.Linear(root_layer_sizes[len(root_layer_sizes) - 1], output_dim))

    def forward(self, x):
        for layer_forward in self.layers:
            x = layer_forward(x)
        return x
class DOD(nn.Module):
    def __init__(self, seed_dim, num_roots, root_output_dim, root_layer_sizes=None):
        super(DOD, self).__init__()
        self.seed_module = StatSeedModule(1, seed_dim)
        self.root_modules = nn.ModuleList([StatRootModule(seed_dim, root_output_dim, root_layer_sizes) for _ in range(num_roots)])

    def forward(self, mu):
        seed_output = self.seed_module(mu)
        root_outputs = [root(seed_output) for root in self.root_modules]
        v_mu_reduced = torch.stack(root_outputs, dim=0).transpose(0, 1)
        return v_mu_reduced

# Define the stationary Coefficient Finder for the stationary equivalent of the problem
class StatPhi1Module(nn.Module):
    def __init__(self, parameter1_dim, m_0, n_0, layer_sizes, leaky_relu_slope=0.1):
        super(StatPhi1Module, self).__init__()
        self.m = m_0
        self.n = n_0
        if layer_sizes is None:
            layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(parameter1_dim, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 1:
                self.layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        self.layers.append(nn.Linear(layer_sizes[len(layer_sizes) - 1], n_0 * m_0))

    def forward(self, x):
        for layer_forward in self.layers:
            x = layer_forward(x)
        x = x.view(-1, self.m, self.n)
        return x
class StatPhi2Module(nn.Module):
    def __init__(self, parameter2_dim, m_0, n_0, layer_sizes, leaky_relu_slope=0.1):
        super(StatPhi2Module, self).__init__()
        self.m = m_0
        self.n = n_0
        if layer_sizes is None:
            layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(parameter2_dim, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 1:
                self.layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        self.layers.append(nn.Linear(layer_sizes[len(layer_sizes) - 1], n_0 * m_0))

    def forward(self, x):
        for layer_forward in self.layers:
            x = layer_forward(x)
        x = x.view(-1, self.m, self.n)
        return x
class CoeffDOD(nn.Module):
    def __init__(self, param_mu_space_dim, param_nu_space_dim, m_0, n_0, layer_sizes=None):
        super(CoeffDOD, self).__init__()
        self.phi_1_module = StatPhi1Module(param_mu_space_dim, m_0, n_0, layer_sizes)
        self.phi_2_module = StatPhi2Module(param_nu_space_dim, m_0, n_0, layer_sizes)

    def forward(self, mu, nu):
        phi_1 = self.phi_1_module(mu)
        phi_2 = self.phi_2_module(nu)
        phi = phi_1 * phi_2
        phi_sum = torch.sum(phi, dim=1).squeeze()
        return phi_sum

# Define the trainer for these
class DODTrainer:
    def __init__(self, nn_model, ambient_dim, train_valid_set, example_name, epochs=1, restart=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.restarts = restart
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = nn_model.to(device)
        self.device = device
        self.N_A = ambient_dim
        self.A = torch.tensor(np.load(f'examples/{example_name}/training_data/stationary_ambient_matrix_{example_name}.npy', allow_pickle=True), dtype=torch.float32).to(self.device)
        self.G = torch.tensor(np.load(f'examples/{example_name}/training_data/stationary_gram_matrix_{example_name}.npy', allow_pickle=True), dtype=torch.float32).to(self.device)

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(StatReducedDatasetLoader(train_data, self.G, self.A, self.N_A), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(StatReducedDatasetLoader(valid_data, self.G, self.A, self.N_A), batch_size=self.batch_size, shuffle=False)


    def loss_function(self, mu_batch, solution_batch):
        batch_size = mu_batch.size(0)  # Get the batch size (can be problem, if set is non-divisible)

        output = self.model(mu_batch)

        # Reshape solution_batch to a 3D tensor with shape [batch_size, N_A, 1]
        solution_batch = solution_batch.unsqueeze(2)

        # Perform batch matrix multiplications
        v_u = torch.bmm(output, solution_batch)
        u_proj = torch.bmm(output.transpose(1, 2), v_u)

        error = solution_batch - u_proj
        loss = torch.sum(torch.norm(error, dim=1) ** 2) / batch_size

        return loss

    def train(self):
        best_model = None
        best_loss = float('inf')

        for _ in range(self.restarts):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                for mu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Model: stationary_DOD, Restart: {_ + 1}, Epoch: {epoch} Loss: {total_loss / len(self.train_loader)}")

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for mu_batch, solution_batch in self.valid_loader:
                    val_loss += self.loss_function(mu_batch, solution_batch).item()
                val_loss /= len(self.valid_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()

            print(f'Restart stationary_DOD. Gen Count at {_ + 1} with current best loss {best_loss}')

        # Load the best model
        self.model.load_state_dict(best_model)
        return best_loss
class CoeffDODTrainer:
    def __init__(self, dod_model, coeffnn_model, ambient_dim, train_valid_set, example_name, epochs=1, restarts=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.dod = dod_model.to(device)
        self.model = coeffnn_model.to(device)
        self.device = device
        self.N_A = ambient_dim

        A = torch.tensor(np.load(f'examples/{example_name}/training_data/stationary_ambient_matrix_{example_name}.npy', allow_pickle=True), dtype=torch.float32).to(self.device)
        G = torch.tensor(np.load(f'examples/{example_name}/training_data/stationary_gram_matrix_{example_name}.npy', allow_pickle=True), dtype=torch.float32).to(self.device)

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)

        # Pre-expand A and G for the batch size
        self.A_expanded = A.unsqueeze(0).expand(self.batch_size, -1, -1)
        self.G_expanded = G.unsqueeze(0).expand(self.batch_size, -1, -1)

    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)  # Get the batch size (can be problem, if set is non-divisible)

        output = self.model(mu_batch, nu_batch)
        dod_output = self.dod(mu_batch)

        # Adjust pre-expanded matrices to the current batch size
        A_expanded = self.A_expanded[:batch_size]
        G_expanded = self.G_expanded[:batch_size]

        # Reshape solution_batch to a 3D tensor with shape [batch_size, N_h, 1]
        solution_batch = solution_batch.unsqueeze(2)

        # Perform batch matrix multiplications
        v_transposed = torch.bmm(dod_output, A_expanded.transpose(1, 2))
        u_proj = torch.bmm(v_transposed, torch.bmm(G_expanded, solution_batch)).squeeze(2)

        error = output - u_proj
        loss = torch.sum(torch.norm(error, dim=1) ** 2) / batch_size

        return loss

    def train(self):
        best_model = None
        best_loss = float('inf')

        for _ in range(self.restarts):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Model: stationary_CoeffDOD, Restart: {_ + 1}, Epoch: {epoch} Loss: {total_loss / len(self.train_loader)}")

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for mu_batch, nu_batch, solution_batch in self.valid_loader:
                    val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                val_loss /= len(self.valid_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()

            print(f'Restart stationary Coefficient Finder. Gen Count at {_ + 1} with current best loss {best_loss}')

        # Load the best model
        self.model.load_state_dict(best_model)
        return best_loss

# Define Hyper Network for time and physical parameter
class Alpha(nn.Module):
    def __init__(self, physical_dim):
        super(Alpha, self).__init__()
        self.fc = nn.Linear(physical_dim + 1, 1, bias=True)

    def forward(self, nu, t):
        # Convert t to a tensor if it's not already one
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=nu.dtype, device=nu.device)
        # Ensure t has a proper shape for concatenation
        input_tensor = torch.cat((nu, t.expand(nu.shape[0], 1)), dim=-1)
        out = self.fc(input_tensor)
        return out

# Define CoLoRA total module 
# takes [B, N_A] \times [B, Theta'] \times [B, 1]
# yields [B, N_A]
class CoLoRA_DL(nn.Module):
    def __init__(self, out_dim, L, dyn_dim, physical_dim, with_bias=True):
        super(CoLoRA_DL, self).__init__()
        self.out_dim = out_dim
        self.L = L
        self.dyn_dim = dyn_dim
        self.with_bias = with_bias

        self.w_init = nn.init.kaiming_normal_
        self.z_init = nn.init.zeros_

        self.Ws = nn.ParameterList([
            nn.Parameter(torch.empty((self.out_dim, self.out_dim), dtype=torch.float32))
            for _ in range(L)
        ])
        self.As = nn.ParameterList([
            nn.Parameter(torch.empty((self.out_dim, self.dyn_dim), dtype=torch.float32))
            for _ in range(L)
        ])
        self.Bs = nn.ParameterList([
            nn.Parameter(torch.empty((self.dyn_dim, self.out_dim), dtype=torch.float32))
            for _ in range(L)
        ])

        self.alphas = nn.ModuleList([Alpha(physical_dim) for _ in range(L)])

        if self.with_bias:
            self.bs = nn.ParameterList([
                nn.Parameter(torch.zeros(self.out_dim, dtype=torch.float32))
                for _ in range(L)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.L):
            self.z_init(self.Ws[i])
            self.w_init(self.As[i])
            self.w_init(self.Bs[i])
            if self.with_bias:
                self.z_init(self.bs[i])

    def forward(self, X, nu, t):
        for i in range(self.L):
            X = X.unsqueeze(2) if X.ndim == 2 else X # assert (B, N_A, 1)
            alpha_val = self.alphas[i](nu, t).unsqueeze(1)  # (B, 1, 1)
            A_exp = self.As[i].unsqueeze(0)  # (1, N_A, dyn_dim)
            AB = (A_exp * alpha_val) @ self.Bs[i]  # (B, N_A, N_A)
            W = self.Ws[i] + AB  # (B, N_A, N_A)
            X = torch.bmm(W, X).squeeze(2)  # (B, N_A, 1) -> (B, N_A)
            if self.with_bias:
                X = X + self.bs[i].unsqueeze(0).expand_as(X)
        return X

# Define CoLoRA trainer
class CoLoRA_DL_Trainer():
    def __init__(self, N_A, DOD_0_model, coeffnn_0_model, colora_model, train_valid_set, example_name, epochs, restarts, learning_rate,
                 batch_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.DOD_0 = DOD_0_model.to(device)
        self.model_0 = coeffnn_0_model.to(device)
        self.colora_model = colora_model.to(device)
        self.device = device

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)
        temp_error = 0.0
        for i in range(nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i * time_end / (nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            # Get the stationary coefficient model output; expected shape: (B, n, 1)
            coeff_0_output = self.model_0(mu_batch, nu_batch).unsqueeze(2)
            # Get the stationary DOD model output; expected shape: (B, n, N_A)
            DOD_0_output = self.DOD_0(mu_batch)
            # Get the CoLoRA prediction output
            output = self.colora_model(torch.bmm(DOD_0_output.transpose(1, 2), coeff_0_output), nu_batch, t_batch)

            # Extract the solution slice at time step i; expected shape: (B, N_A)
            u_proj = solution_batch[:, i, :]

            error = output - u_proj  # (B, N_A)
            temp_error += torch.sum(torch.norm(error, dim=1) ** 2)

        loss = temp_error / batch_size
        return loss
    
    def train(self):
        best_model = None
        best_loss = float('inf')

        for _ in range(self.restarts):
            self.colora_model.apply(initialize_weights)
            optimizer = optim.Adam(self.colora_model.parameters(), lr=self.learning_rate)
            for epoch in range(self.epochs):
                self.colora_model.train()
                total_loss = 0
                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Model: CoLoRA_DL, Restart: {_ + 1}, Epoch: {epoch} Loss: {total_loss / len(self.train_loader)}")

            self.colora_model.eval()
            with torch.no_grad():
                val_loss = 0
                for mu_batch, nu_batch, solution_batch in self.valid_loader:
                    val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                val_loss /= len(self.valid_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.colora_model.state_dict()

            print(f'Restart CoLoRA_DL. Gen Count at {_ + 1} with current best loss {best_loss}')

        # Load the best model
        self.colora_model.load_state_dict(best_model)
        return best_loss

'''
---------------------
Further additions for testing
---------------------
'''

# Works on numpy and torch
class L2Norm():
    def __init__(self, G):
        self.G = G

    def __call__(self, *args, **kwds):
        if isinstance(*args, torch.Tensor):
            return torch.sum(torch.sqrt(torch.einsum('ij,jk,ik->i', *args, self.G, *args)))
        elif isinstance(*args, np.ndarray):
            return np.sum(np.sqrt(np.einsum('ij,jk,ik->i', *args, self.G, *args)))
        else:
            print("Type Error in L2 Norm")
            pass

# Define a mean error Finder
class MeanError():
    def __init__(self, norm):
        self.norm = norm
    def __call__(self, data, **kwds):
        error = 0
        for entry in data:
            error += self.norm(entry)
        return error / len(data)

