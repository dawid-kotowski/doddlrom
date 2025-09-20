import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
import math
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Some Constants
time_end = 1
nt = 10

# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class FetchTrainAndValidSet:
    """
    Loader for compact .npz datasets saved as:
    - mu:       [Ns] float32
    - nu:       [Ns] float32
    - solution: [Ns, Nt, N_A], [Ns, Nt, N]  (instationary) or [Ns, N_A] (stationary)
    """
    def __init__(self, train_to_val_ratio: float, example_name: str, reduction_type: str):
        self.train_to_val_ratio = train_to_val_ratio
        path_npz = f'examples/{example_name}/training_data/{reduction_type}_training_data_{example_name}.npz'

        data = np.load(path_npz)
        self.mu = data['mu']              # [Ns]
        self.nu = data['nu']              # [Ns]
        self.solution = data['solution']  # [Ns, Nt, N_A], [Ns, Nt, N] or [Ns, N_A]

        Ns = self.mu.shape[0]
        idx = np.arange(Ns, dtype=np.int32)
        np.random.shuffle(idx)
        n_train = int(train_to_val_ratio * Ns)
        self.train_idx = idx[:n_train]
        self.valid_idx = idx[n_train:]

    def _tuples(self, idx_array):
        return [(float(self.mu[i]), float(self.nu[i]), self.solution[i]) for i in idx_array]

    def __call__(self, set_type: str):
        if set_type == 'train':
            return self._tuples(self.train_idx)
        elif set_type == 'valid':
            return self._tuples(self.valid_idx)
        else:
            raise ValueError("Type must be 'train' or 'valid'")

# Define DatasetLoader
class DatasetLoader(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mu, nu, solution = self.data[idx]   # unpack tuple
        mu = torch.tensor(mu, dtype=torch.float32).unsqueeze(0)  # keep 2D shape
        nu = torch.tensor(nu, dtype=torch.float32).unsqueeze(0)
        solution = torch.tensor(solution, dtype=torch.float32)
        return mu, nu, solution



''' 
------------------------------------
In the following a innerDOD is introduced, which dynamically approximates a reduced basis
w.r.t the geometric parameter mu

V: (Theta \times [0, T]) -> (R^(N_h \times N'))
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

# Define Root Modules for innerDOD
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
# returns a tensor of size [N_A, N']
class innerDOD(nn.Module):
    def __init__(self, seed_dim, geometric_dim, root_layer_sizes, N_prime, N_A):
        super(innerDOD, self).__init__()
        self.seed_module = SeedModule(geometric_dim, seed_dim)
        self.root_modules = nn.ModuleList(
            [RootModule(seed_dim, N_A, root_layer_sizes) for _ in range(N_prime)])

    def forward(self, mu, t):
        seed_output = self.seed_module(mu)
        mu_t = torch.cat((seed_output, t), dim=1)
        root_outputs = [root(mu_t) for root in self.root_modules]
        v_mu = torch.stack(root_outputs, dim=0).transpose(0,1)
        return v_mu

# Define DOD_DL_-DL training

class innerDODTrainer:
    def __init__(self, dod_model, train_valid_set, epochs=1, restart=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience = 3):
        self.learning_rate = learning_rate
        self.restarts = restart
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = dod_model.to(device)
        self.device = device
        self.patience = patience

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), 
                                       batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), 
                                       batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, solution_batch):
        batch_size = mu_batch.size(0)
        temp_loss = 0.

        for i in range(nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i * time_end/(nt + 1), 
                              dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            output = self.model(mu_batch, t_batch)

            v_u = torch.bmm(output, solution_batch[:, i, :].unsqueeze(2))
            u_proj = torch.bmm(output.transpose(1, 2), v_u)

            error = solution_batch[:, i, :].unsqueeze(2) - u_proj
            temp_loss += torch.sum(torch.norm(error, dim=1) ** 2)

        loss = temp_loss / (batch_size * (nt + 1))
        return loss

    def train(self):
        best_model = None
        best_loss_restart = float('inf')
        best_loss = float('inf')

        tqdm.write("Model inner_DOD is being trained...")

        for restart_idx in tqdm(range(self.restarts), desc="Restarts inner_DOD", leave=False):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            epochs_no_improve = 0
            best_loss_restart = float('inf')

            for epoch in tqdm(range(self.epochs), desc=f"Epochs [Restart {restart_idx + 1}]", leave=False):
                self.model.train()
                total_loss = 0

                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for mu_batch, nu_batch, solution_batch in self.valid_loader:
                        val_loss += self.loss_function(mu_batch, solution_batch).item()
                    val_loss /= len(self.valid_loader)

                if val_loss < best_loss_restart:
                    best_loss_restart = val_loss
                    best_model = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        tqdm.write(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break

                tqdm.write(f"Restart: {restart_idx + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.6f}")


            if best_loss_restart < best_loss:
                best_model = self.model.state_dict()
                best_loss = best_loss_restart

            tqdm.write(f'Current best loss of inner_DOD: {best_loss:.6f}')

        self.model.load_state_dict(best_model)
        return best_loss

''' 
------------------------------------
Adding to the innerDOD is a network trying to approximate the latent dynamics of the underlying 
n-dim solution manifold. We present the following different approaches for this

DOD+DFNN     : (Theta \times Theta' \times [0, T]) -> R^N'
             ; Linear(Linear(mu_t)) @ Linear(Linear(nu_t)) \mapsto u_N'

DOD-DL-ROM   : (Theta \times Theta' \times [0, T]) -> R^N'
             ; Encoder(Linear(Linear(mu_t)) @ Linear(Linear(nu_t))) \mapsto u_N'
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

# Define Complete parameter-to-innerDOD-coefficients Model 
# returns [B, N']
class HadamardNN(nn.Module):
    def __init__(self, geometric_dim, physical_dim, m_0, n_0, layer_sizes=None):
        super(HadamardNN, self).__init__()
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

# Define optional parameter-to-latent-dynamic Model
#returns [B, N'] or [B, n]
class DFNN(nn.Module):
    def __init__(self, geometric_dim, physical_dim, n, layer_sizes=None, leaky_relu_slope=0.1):
        super(DFNN, self).__init__()
        if layer_sizes is None:
            layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(geometric_dim + physical_dim + 1, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 1:
                self.layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
        self.layers.append(nn.Linear(layer_sizes[len(layer_sizes) - 1], n))
    def forward(self, mu, nu, t):
        mu_nu_t = torch.cat((mu, nu, t), dim=1)
        for layer_forward in self.layers:
            mu_nu_t = layer_forward(mu_nu_t)
        return mu_nu_t

# Define the Trainer for both HadamardNN and DFNN
class DFNNTrainer:
    def __init__(self, N_A, DOD_DL_model, coeffnn_model, train_valid_set, epochs, restarts, learning_rate,
                 batch_size, device='cuda' if torch.cuda.is_available() else 'cpu', patience = 3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.innerDOD = DOD_DL_model.to(device)
        self.model = coeffnn_model.to(device)
        self.device = device
        self.patience = patience

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
            DOD_DL_output = self.innerDOD(mu_batch, t_batch)

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
        best_loss_restart = float('inf')
        best_loss = float('inf')

        tqdm.write("Model Coeff DOD DL is being trained...")

        for restart_idx in tqdm(range(self.restarts), desc="Restarts Coeff DOD DL", leave=False):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            epochs_no_improve = 0
            best_loss_restart = float('inf')

            for epoch in tqdm(range(self.epochs), desc=f"Epochs [Restart {restart_idx + 1}]", leave=False):
                self.model.train()
                total_loss = 0

                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for mu_batch, nu_batch, solution_batch in self.valid_loader:
                        val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                    val_loss /= len(self.valid_loader)

                if val_loss < best_loss_restart:
                    best_loss_restart = val_loss
                    best_model = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        tqdm.write(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break

                tqdm.write(f"Restart: {restart_idx + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.6f}")


            if best_loss_restart < best_loss:
                best_model = self.model.state_dict()
                best_loss = best_loss_restart

            tqdm.write(f'Current best loss of Coeff DOD DL: {best_loss:.6f}')

        self.model.load_state_dict(best_model)
        return best_loss
    
# Define Encoder 
# takes [B, input_dim] return [B, N' = loop(floor((input_dim + 2p - k) / s) + 1)**2)] 
class Encoder(nn.Module):
    def __init__(self, input_dim, in_channels, hidden_channels, latent_dim,
                num_layers, kernel, stride, padding):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.padding = padding
        self.kernel = kernel
        self.stride = stride
        self.encoder_output_shape = self._compute_encoder_shape() 
        self.grid_size = self._compute_grid(input_dim)
        H, W = self.grid_size

        # Convolutional layers
        conv_layers = []
        current_channels = in_channels
        for i in range(num_layers):
            next_channels = hidden_channels
            conv_layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=kernel,
                                    stride=stride, padding=padding))
            conv_layers.append(nn.LeakyReLU(0.1))
            current_channels = next_channels
        self.conv = nn.Sequential(*conv_layers)

        # Linear layers
        linear_layers = []
        C, H, W = self.encoder_output_shape
        current_dim = int(C*H*W)
        for i in range(self._compute_linear_num() - 1):
            linear_layers.append(nn.Linear(int(current_dim), int(current_dim // 2)))
            linear_layers.append(nn.LeakyReLU(0.1))
            current_dim = current_dim // 2
        linear_layers.append(nn.Linear(int(current_dim), int(self.latent_dim)))

        self.linear = nn.Sequential(*linear_layers)

    def _compute_grid(self, D):
        size = math.ceil(math.sqrt(D))
        return (size, size)
    
    def _compute_linear_num(self):
        C, H, W = self._compute_encoder_shape()
        return int(np.floor(C*H*W / self.latent_dim)) - 1

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
        conv_out = self.conv(x2d).view(B, -1)
        z = self.linear(conv_out) 
        return z

# Define Decoder 
# takes [B, N' = loop(floor((input_dim + 2p - k) / s) + 1)**2)] return [B, input_dim]
class Decoder(nn.Module):
    def __init__(self, output_dim, out_channels, hidden_channels, latent_dim, 
                 num_layers, kernel, stride, padding):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.padding = padding
        self.kernel = kernel
        self.stride = stride
        self.encoder_output_shape = self._compute_encoder_shape()
        C, H, W = self._compute_encoder_shape()

        # Linear layers
        linear_layers = []
        current_dim = self.latent_dim
        for i in range(self._compute_linear_num() - 1):
            linear_layers.append(nn.Linear(int(current_dim), int(current_dim * 2)))
            linear_layers.append(nn.LeakyReLU(0.1))
            current_dim = current_dim * 2
        linear_layers.append(nn.Linear(int(current_dim), int(C*H*W)))
        self.delinear = nn.Sequential(*linear_layers)

        # Convolutional layers
        conv_layers = []
        current_channels = C
        for i in range(num_layers):
            next_channels = hidden_channels if i < num_layers - 1 else out_channels
            conv_layers.append(nn.ConvTranspose2d(current_channels, next_channels, 
                                             kernel_size=kernel, stride=stride, padding=padding, output_padding=1))
            if i < num_layers - 1:
                conv_layers.append(nn.LeakyReLU(0.1))
            current_channels = next_channels
        self.deconv = nn.Sequential(*conv_layers)

    def _compute_linear_num(self):
        C, H, W = self._compute_encoder_shape()
        return int(np.floor(C*H*W / self.latent_dim)) - 1
    
    def _compute_encoder_shape(self):
        D_temp = math.ceil(math.sqrt(self.output_dim))
        for layer in range(self.num_layers):
            D_temp = math.floor((D_temp + 2 * self.padding - self.kernel) / self.stride) + 1
        return (self.hidden_channels, D_temp, D_temp)

    def forward(self, z):
        B = z.shape[0]
        x_unflat = self.delinear(z)
        x_unflat = x_unflat.view(B, *self.encoder_output_shape)
        x_recon = self.deconv(x_unflat)
        return x_recon.view(B, -1)[:, :self.output_dim]
    
# Define the Trainer for the DOD-DL-ROM Coefficients
class DOD_DL_ROMTrainer:
    def __init__(self, DOD_DL_model, Coeff_DOD_DL_model, Encoder_model, Decoder_model, train_valid_set, error_weight=0.5,
                 epochs=1, restarts=1, learning_rate=1e-3, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.learning_rate = learning_rate
        self.error_weight = error_weight
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.innerDOD = DOD_DL_model.to(device)
        self.en_model = Encoder_model.to(device)
        self.de_model = Decoder_model.to(device)
        self.coeff_model = Coeff_DOD_DL_model.to(device)
        self.device = device
        self.patience = patience

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
            DOD_DL_output = self.innerDOD(mu_batch, t_batch)

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
        best_loss_restart = float('inf')

        tqdm.write("Model AE DOD DL is being trained...")


        for restart_idx in tqdm(range(self.restarts), desc="Restarts AE DOD DL", leave=False):
            self.coeff_model.apply(initialize_weights)
            self.en_model.apply(initialize_weights)
            self.de_model.apply(initialize_weights)
            
            params = list(self.coeff_model.parameters()) + \
                     list(self.en_model.parameters()) + \
                     list(self.de_model.parameters())
            optimizer = optim.Adam(params, lr=self.learning_rate)

            epochs_no_improve = 0
            best_loss_restart = float('inf')

            
            for epoch in tqdm(range(self.epochs), desc=f"Epochs [Restart {restart_idx + 1}]", leave=False):
                self.coeff_model.train()
                self.en_model.train()
                self.de_model.train()
                self.innerDOD.train()  
                total_loss = 0.0

                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                self.coeff_model.eval()
                self.en_model.eval()
                self.de_model.eval()
                self.innerDOD.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for mu_batch, nu_batch, solution_batch in self.valid_loader:
                        val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                    val_loss /= len(self.valid_loader)
            
                if val_loss < best_loss_restart:
                    best_loss_restart = val_loss
                    best_model = {
                        "encoder": copy.deepcopy(self.en_model.state_dict()),
                        "decoder": copy.deepcopy(self.de_model.state_dict()),
                        "coeff_model": copy.deepcopy(self.coeff_model.state_dict())
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        tqdm.write(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break


                tqdm.write(f"Restart: {restart_idx + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.6f}")

            if best_loss_restart < best_loss:
                best_model = {
                        "encoder": copy.deepcopy(self.en_model.state_dict()),
                        "decoder": copy.deepcopy(self.de_model.state_dict()),
                        "coeff_model": copy.deepcopy(self.coeff_model.state_dict())
                    }
                best_loss = best_loss_restart

            tqdm.write(f'Current best loss of AE DOD DL: {best_loss:.6f}')

        if best_model is not None:
            self.en_model.load_state_dict(best_model["encoder"])
            self.de_model.load_state_dict(best_model["decoder"])
            self.coeff_model.load_state_dict(best_model["coeff_model"])
            
        return best_loss

'''
-----------------------------------
We furthermore add a POD DL ROM Trainer on the basis of the Autoencoder,
as the POD Matrix is already provided. The approximation is then given via

u(mu, nu, t) = A Decoder(Coeff(mu, nu, t))
-----------------------------------
'''
# Define the Trainer for the standard POD DL
class POD_DL_ROMTrainer:
    def __init__(self, Coeff_model, Encoder_model, Decoder_model, train_valid_set, error_weight,
                 epochs=1, restarts=1, learning_rate=1e-3, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.error_weight = error_weight
        self.restarts = restarts
        self.batch_size = batch_size
        self.en_model = Encoder_model.to(device)
        self.de_model = Decoder_model.to(device)
        self.coeff_model = Coeff_model.to(device)
        self.device = device
        self.patience = patience

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
            # Get the non linear expansion output; expected shape: (B, N_A)
            decoder_output = self.de_model(coeff_output)

            # Extract the solution slice at time step i; expected shape: (B, N_A)
            solution_slice = solution_batch[:, i, :]

            # Encoder output; expected shape: (B, N)
            encoder_output = self.en_model(solution_slice)

            dynam_error = solution_slice - decoder_output  # (B, N_A)
            proj_error = encoder_output - coeff_output # (B, n)
            temp_error += (self.error_weight / 2 * torch.sum(torch.norm(dynam_error, dim=1) ** 2) 
                           + (1-self.error_weight) / 2 * torch.sum(torch.norm(proj_error, dim=1) ** 2))

        loss = temp_error / batch_size
        return loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        best_loss_restart = float('inf')

        tqdm.write("Model POD DL is being trained...")


        for restart_idx in tqdm(range(self.restarts), desc="Restarts POD DL", leave=False):
            self.coeff_model.apply(initialize_weights)
            self.en_model.apply(initialize_weights)
            self.de_model.apply(initialize_weights)
            
            params = list(self.coeff_model.parameters()) + \
                     list(self.en_model.parameters()) + \
                     list(self.de_model.parameters())
            optimizer = optim.Adam(params, lr=self.learning_rate)

            epochs_no_improve = 0
            best_loss_restart = float('inf')
            
            for epoch in tqdm(range(self.epochs), desc=f"Epochs [Restart {restart_idx + 1}]", leave=False):
                self.coeff_model.train()
                self.en_model.train()
                self.de_model.train()
                total_loss = 0.0

                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                self.coeff_model.eval()
                self.en_model.eval()
                self.de_model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for mu_batch, nu_batch, solution_batch in self.valid_loader:
                        val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                    val_loss /= len(self.valid_loader)
            
                if val_loss < best_loss_restart:
                    best_loss_restart = val_loss
                    best_model = {
                        "encoder": copy.deepcopy(self.en_model.state_dict()),
                        "decoder": copy.deepcopy(self.de_model.state_dict()),
                        "coeff_model": copy.deepcopy(self.coeff_model.state_dict())
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        tqdm.write(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break


                tqdm.write(f"Restart: {restart_idx + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.6f}")

            if best_loss_restart < best_loss:
                best_model = {
                        "encoder": copy.deepcopy(self.en_model.state_dict()),
                        "decoder": copy.deepcopy(self.de_model.state_dict()),
                        "coeff_model": copy.deepcopy(self.coeff_model.state_dict())
                    }
                best_loss = best_loss_restart

            tqdm.write(f'Current best loss of POD DL: {best_loss:.6f}')

        if best_model is not None:
            self.en_model.load_state_dict(best_model["encoder"])
            self.de_model.load_state_dict(best_model["decoder"])
            self.coeff_model.load_state_dict(best_model["coeff_model"])
            
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
class statDOD(nn.Module):
    def __init__(self, geometric_dim, seed_dim, num_roots, root_output_dim, root_layer_sizes=None):
        super(statDOD, self).__init__()
        self.seed_module = StatSeedModule(geometric_dim, seed_dim)
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
class statHadamardNN(nn.Module):
    def __init__(self, param_mu_space_dim, param_nu_space_dim, m_0, n_0, layer_sizes=None):
        super(statHadamardNN, self).__init__()
        self.phi_1_module = StatPhi1Module(param_mu_space_dim, m_0, n_0, layer_sizes)
        self.phi_2_module = StatPhi2Module(param_nu_space_dim, m_0, n_0, layer_sizes)

    def forward(self, mu, nu):
        phi_1 = self.phi_1_module(mu)
        phi_2 = self.phi_2_module(nu)
        phi = phi_1 * phi_2
        phi_sum = torch.sum(phi, dim=1).squeeze()
        return phi_sum

# Define the trainer for these
class statDODTrainer:
    def __init__(self, nn_model, ambient_dim, train_valid_set, epochs=1, restart=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.learning_rate = learning_rate
        self.restarts = restart
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = nn_model.to(device)
        self.device = device
        self.N_A = ambient_dim
        self.patience = patience

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)


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
        best_loss_restart = float('inf')
        best_loss = float('inf')

        tqdm.write("Model stat DOD is being trained...")

        for restart_idx in tqdm(range(self.restarts), desc="Restarts stat DOD", leave=False):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            epochs_no_improve = 0
            best_loss_restart = float('inf')

            for epoch in tqdm(range(self.epochs), desc=f"Epochs [Restart {restart_idx + 1}]", leave=False):
                self.model.train()
                total_loss = 0

                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for mu_batch, nu_batch, solution_batch in self.valid_loader:
                        val_loss += self.loss_function(mu_batch, solution_batch).item()
                    val_loss /= len(self.valid_loader)

                if val_loss < best_loss_restart:
                    best_loss_restart = val_loss
                    best_model = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        tqdm.write(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break

                tqdm.write(f"Restart: {restart_idx + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.6f}")


            if best_loss_restart < best_loss:
                best_model = self.model.state_dict()
                best_loss = best_loss_restart

            tqdm.write(f'Current best loss of stat DOD: {best_loss:.6f}')

        self.model.load_state_dict(best_model)
        return best_loss
class statHadamardNNTrainer:
    def __init__(self, dod_model, coeffnn_model, ambient_dim, train_valid_set, epochs=1, restarts=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.dod = dod_model.to(device)
        self.model = coeffnn_model.to(device)
        self.device = device
        self.N_A = ambient_dim
        self.patience = patience

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)


    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)
        coeff_output = self.model(mu_batch, nu_batch)
        dod_output = self.dod(mu_batch)

        # Reshape solution_batch to a 3D tensor with shape [batch_size, N_A, 1]
        solution_batch = solution_batch.unsqueeze(2)

        # Perform batch matrix multiplications
        output = torch.bmm(dod_output, solution_batch).squeeze(2)
        error = output - coeff_output
        loss = torch.sum(torch.norm(error, dim=1) ** 2) / batch_size

        return loss

    def train(self):
        best_model = None
        best_loss_restart = float('inf')
        best_loss = float('inf')

        tqdm.write("Model stat Coeff DOD is being trained...")

        for restart_idx in tqdm(range(self.restarts), desc="Restarts stat Coeff DOD", leave=False):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            epochs_no_improve = 0
            best_loss_restart = float('inf')

            for epoch in tqdm(range(self.epochs), desc=f"Epochs [Restart {restart_idx + 1}]", leave=False):
                self.model.train()
                total_loss = 0

                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for mu_batch, nu_batch, solution_batch in self.valid_loader:
                        val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                    val_loss /= len(self.valid_loader)

                if val_loss < best_loss_restart:
                    best_loss_restart = val_loss
                    best_model = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        tqdm.write(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break

                tqdm.write(f"Restart: {restart_idx + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.6f}")


            if best_loss_restart < best_loss:
                best_model = self.model.state_dict()
                best_loss = best_loss_restart

            tqdm.write(f'Current best loss of stat Coeff DOD: {best_loss:.6f}')

        self.model.load_state_dict(best_model)
        return best_loss

# Define Hyper Network for time and physical parameter
class Alpha(nn.Module):
    def __init__(self, physical_dim):
        super(Alpha, self).__init__()
        self.fc = nn.Linear(physical_dim + 1, 1, bias=True)

    def forward(self, nu, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=nu.dtype, device=nu.device)
        # Ensure t has a proper shape for concatenation
        input_tensor = torch.cat((nu, t.expand(nu.shape[0], 1)), dim=-1)
        out = self.fc(input_tensor)
        return out

# Define CoLoRA total module 
# takes [B, N_A] \times [B, Theta'] \times [B, 1]
# yields [B, N_A]
class CoLoRA(nn.Module):
    def __init__(self, out_dim, L, dyn_dim, physical_dim, with_bias=True):
        super(CoLoRA, self).__init__()
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
class CoLoRATrainer():
    def __init__(self, DOD_0_model, coeffnn_0_model, colora_model, train_valid_set, epochs, restarts, learning_rate,
                 batch_size, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.DOD_0 = DOD_0_model.to(device)
        self.model_0 = coeffnn_0_model.to(device)
        self.model = colora_model.to(device)
        self.device = device
        self.patience = patience

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
            output = self.model(torch.bmm(DOD_0_output.transpose(1, 2), coeff_0_output), nu_batch, t_batch)

            # Extract the solution slice at time step i; expected shape: (B, N_A)
            u_proj = solution_batch[:, i, :]

            error = output - u_proj  # (B, N_A)
            temp_error += torch.sum(torch.norm(error, dim=1) ** 2)

        loss = temp_error / batch_size
        return loss
    
    def train(self):
        best_model = None
        best_loss_restart = float('inf')
        best_loss = float('inf')

        tqdm.write("Model CoLoRA is being trained...")

        for restart_idx in tqdm(range(self.restarts), desc="Restarts CoLoRA DL", leave=False):
            self.model.apply(initialize_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

            epochs_no_improve = 0
            best_loss_restart = float('inf')

            for epoch in tqdm(range(self.epochs), desc=f"Epochs [Restart {restart_idx + 1}]", leave=False):
                self.model.train()
                total_loss = 0

                for mu_batch, nu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, nu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for mu_batch, nu_batch, solution_batch in self.valid_loader:
                        val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                    val_loss /= len(self.valid_loader)

                if val_loss < best_loss_restart:
                    best_loss_restart = val_loss
                    best_model = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        tqdm.write(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                        break

                tqdm.write(f"Restart: {restart_idx + 1}, Epoch: {epoch + 1}, Val Loss: {val_loss:.6f}")


            if best_loss_restart < best_loss:
                best_model = self.model.state_dict()
                best_loss = best_loss_restart

            tqdm.write(f'Current best loss of CoLoRA DL: {best_loss:.6f}')

        self.model.load_state_dict(best_model)
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

'''
-------------------
Define simple forward pass assuming existence of networks
-------------------
'''

def dod_dfnn_forward(A, innerDOD_model, Coeff_model, mu_i, nu_i, nt_):
    dod_dl_solution = []
    for j in range(nt_ + 1):
        time = torch.tensor(j * time_end / (nt + 1), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        time = time.unsqueeze(0).unsqueeze(1)
        dod_dl_output = innerDOD_model(mu_i, time).squeeze(0).T
        coeff_output = Coeff_model(mu_i, nu_i, time).squeeze(0)
        u_ij_coeff_dl = torch.matmul(torch.matmul(A, dod_dl_output), coeff_output)
        dod_dl_solution.append(u_ij_coeff_dl)

    u_i_dod_dl = torch.stack(dod_dl_solution, dim=0)
    u_i_dod_dl = u_i_dod_dl.detach().numpy()
    return u_i_dod_dl

def pod_dl_rom_forward(A_P, POD_DL_model, De_model, mu_i, nu_i, nt_):
    pod_dl_solution = []
    for j in range(nt_ + 1):
        time = torch.tensor(j * time_end / (nt + 1), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        time = time.unsqueeze(0).unsqueeze(1)
        coeff_n_output = POD_DL_model(mu_i, nu_i, time).unsqueeze(0)
        decoded_output = De_model(coeff_n_output).squeeze(0)
        u_ij_pod_dl = torch.matmul(A_P, decoded_output)
        pod_dl_solution.append(u_ij_pod_dl)
    
    u_i_pod_dl = torch.stack(pod_dl_solution, dim=0)
    u_i_pod_dl = u_i_pod_dl.detach().numpy()
    return u_i_pod_dl

def dod_dl_rom_forward(A, innerDOD_model, DOD_DL_model, De_model, mu_i, nu_i, nt_):
    dod_dl_rom_solution = []
    for j in range(nt_ + 1):
        time = torch.tensor(j * time_end / (nt + 1), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        time = time.unsqueeze(0).unsqueeze(1)
        dod_dl_output = innerDOD_model(mu_i, time).squeeze(0).T
        coeff_n_output = DOD_DL_model(mu_i, nu_i, time)
        decoded_output = De_model(coeff_n_output).squeeze(0)
        u_ij_dod_dl = torch.matmul(torch.matmul(A, dod_dl_output), decoded_output)
        dod_dl_rom_solution.append(u_ij_dod_dl)
    
    u_i_dod_dl_rom = torch.stack(dod_dl_rom_solution, dim=0)
    u_i_dod_dl_rom = u_i_dod_dl_rom.detach().numpy()
    return u_i_dod_dl_rom

def colora_forward(A, stat_DOD_model, stat_Coeff_model, CoLoRA_DL_model, mu_i, nu_i, nt_):
    colora_dl_solution = []
    for j in range(nt_ + 1):
        time = torch.tensor(j * time_end / (nt + 1), dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        time = time.unsqueeze(0).unsqueeze(1)
        stat_coeff_n_output = stat_Coeff_model(mu_i, nu_i).unsqueeze(0).unsqueeze(2)
        stat_dod_output = stat_DOD_model(mu_i)
        v_0 = torch.bmm(stat_dod_output.transpose(1, 2), stat_coeff_n_output)
        u_ij_colora = torch.matmul(A, CoLoRA_DL_model(v_0, nu_i, time).squeeze(0))
        colora_dl_solution.append(u_ij_colora)

    u_i_colora_dl = torch.stack(colora_dl_solution, dim=0)
    u_i_colora_dl = u_i_colora_dl.detach().numpy()
    return u_i_colora_dl