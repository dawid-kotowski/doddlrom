import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
import math
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
    Loads compact .npz saved as:
      - mu:       [Ns] float32
      - nu:       [Ns] float32
      - solution: [Ns, Nt, D]  (instationary) or [Ns, D] (stationary)
    Normalizes mu, nu (z-score on TRAIN split) and solution per mode (z-score on TRAIN split).
    D refers to reduction dimension.
    """
    def __init__(self, train_to_val_ratio: float, example_name: str, reduction_tag: str):
        path_npz = f'examples/{example_name}/training_data/{reduction_tag}_training_data_{example_name}.npz'
        data = np.load(path_npz)

        # raw arrays
        self.mu = data['mu'].astype(np.float32)              # [Ns]
        self.nu = data['nu'].astype(np.float32)              # [Ns]
        self.solution = data['solution'].astype(np.float32)  # [Ns, Nt, D] or [Ns, D]

        Ns = self.mu.shape[0]
        idx = np.arange(Ns, dtype=np.int32)
        np.random.shuffle(idx)
        n_train = int(train_to_val_ratio * Ns)
        self.train_idx = idx[:n_train]
        self.valid_idx = idx[n_train:]

        mu_tr = self.mu[self.train_idx]
        nu_tr = self.nu[self.train_idx]
        sol_tr = self.solution[self.train_idx]

        eps = 1e-8
        if sol_tr.ndim == 3:        # [Ntr, Nt, D]
            axis = (0, 1)
        elif sol_tr.ndim == 2:      # [Ntr, D]
            axis = 0
        else:
            raise ValueError(f"Unexpected solution ndim: {sol_tr.ndim}")

        mu_min = float(mu_tr.min()); mu_max = float(mu_tr.max())
        nu_min = float(nu_tr.min()); nu_max = float(nu_tr.max())
        sol_min = sol_tr.min(axis=axis); sol_max = sol_tr.max(axis=axis) # [D]


        # Save 
        stats_path = f'examples/{example_name}/training_data/normalization_{reduction_tag}_{example_name}.npz'
        np.savez_compressed(stats_path,
                            mu_min=mu_min, mu_max=mu_max,
                            nu_min=nu_min, nu_max=nu_max,
                            sol_min=sol_min.astype(np.float32),
                            sol_max=sol_max.astype(np.float32))

        self.mu_n = (self.mu - mu_min) / (mu_max - mu_min + eps)
        self.nu_n = (self.nu - nu_min) / (nu_max - nu_min + eps)
        self.solution_n = (self.solution - sol_min[None,...]) / (
            sol_max[None,...] - sol_min[None,...] + eps)

    def _tuples(self, idx_array):
        return [(float(self.mu_n[i]), float(self.nu_n[i]), self.solution_n[i]) for i in idx_array]

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
        mu, nu, solution = self.data[idx] 
        mu = torch.tensor([mu], dtype=torch.float32)
        nu = torch.tensor([nu], dtype=torch.float32)
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
            """
            Inputs:
            mu: [B, geometric_dim] or [geometric_dim]
            t : [B, t_dim] or [t_dim]
            Returns:
            V: [B, N_A, N'] if batched, else [N_A, N']
            """
            if mu.dim() == 1:
                mu = mu.unsqueeze(0)                                   # [1, geometric_dim]
            if t.dim() == 1:
                t = t.unsqueeze(0)                                     # [1, t_dim]
            B = mu.size(0)
            seed_out = self.seed_module(mu)                            # [B, seed_dim]
            mu_t = torch.cat([seed_out, t], dim=-1)

            root_outputs = [root(mu_t) for root in self.root_modules]  # N' items of [B, N_A]
            V = torch.stack(root_outputs, dim=-1)                      # [B, N_A, N']
            return V if B > 1 else V.squeeze(0)

# Define DOD_DL_-DL training
class innerDODTrainer:
    def __init__(self, nt, dod_model, train_valid_set, epochs=1, restart=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience = 3):
        self.nt = nt
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
        
    def orth_penalty(self, V):
        """
        V: [B, N_A, N']  (columns should be orthonormal in Euclidean sense)
        Returns: scalar penalty
        """
        Bt, NA, Np = V.shape
        S = torch.bmm(V.transpose(1, 2), V)                    # [B, N', N']
        I = torch.eye(Np, device=V.device, dtype=V.dtype).expand_as(S)
        return ((S - I).pow(2).sum(dim=(-2, -1)) / Np).mean()  

    def loss_function(self, mu_batch, solution_batch, lambda_orth=1e-2):
        """
        mu_batch:        [B, geometric_dim]
        solution_batch:  [B, nt+1, N_A]   (solutions already in the ambient space R^{N_A})
        Returns scalar loss (Euclidean, G-independent).
        """
        B = mu_batch.size(0)
        assert solution_batch.dim() == 3, "solution_batch must be [B, nt+1, N_A]"

        temp_proj = 0.0
        temp_orth = 0.0

        for i in range(self.nt + 1):
            t_batch = torch.full((B, 1), i  / (self.nt + 1),
                            dtype=torch.float32, device=self.device) # [B, 1]

            V = self.model(mu_batch, t_batch)                        # [B, N_A, N']
            u = solution_batch[:, i, :].unsqueeze(-1)                # [B, N_A, 1]
            alpha = torch.bmm(V.transpose(1, 2), u)                  # [B, N', 1]
            u_proj = torch.bmm(V, alpha)                             # [B, N_A, 1]

            err = u - u_proj
            temp_proj = temp_proj + torch.sum(torch.norm(err, dim=1) ** 2)

            # orthonormality penalty
            if lambda_orth > 0.0:
                pen = self.orth_penalty(V)
                temp_orth = temp_orth + pen
            
        data_loss = temp_proj / (B * (self.nt + 1))
        orth_loss = (temp_orth / (self.nt + 1))
        loss = data_loss + lambda_orth * orth_loss
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
    def __init__(self, nt, N_A, DOD_DL_model, coeffnn_model, train_valid_set, epochs, restarts, learning_rate,
                 batch_size, device='cuda' if torch.cuda.is_available() else 'cpu', patience = 3):
        self.nt = nt
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
        """
        Dimensionality:
        V = innerDOD(mu,t):           [B, N_A, N']        # Euclidean-orthonormal columns
        coeff_model(mu,nu,t):         [B, N']             # predicted coefficients
        solution_batch[:, i, :]:      [B, N_A]            # ambient POD solution at time i
        alpha_true = V^T u:           [B, N']             # target coefficients
        """
        B = mu_batch.size(0)
        temp_error = 0.0

        for i in range(self.nt + 1):
            t_batch = torch.full((B, 1), i / (self.nt + 1),
                                dtype=torch.float32, device=self.device)                   # [B,1]

            coeff_pred = self.model(mu_batch, nu_batch, t_batch)                             # [B, N']
            V = self.innerDOD(mu_batch, t_batch)                                             # [B, N_A, N']

            u = solution_batch[:, i, :].unsqueeze(-1)                                        # [B, N_A, 1]
            alpha_true = torch.bmm(V.transpose(1, 2), u).squeeze(-1)                         # [B, N']  = (V^T u)

            error = coeff_pred - alpha_true                                                  # [B, N']
            temp_error = temp_error + torch.sum(torch.norm(error, dim=1) ** 2)               

        loss = temp_error / (B * (self.nt + 1))
        return loss


    def train(self):
        best_model = None
        best_loss_restart = float('inf')
        best_loss = float('inf')

        tqdm.write("Model DFNN is being trained...")

        for restart_idx in tqdm(range(self.restarts), desc="Restarts DFNN", leave=False):
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

            tqdm.write(f'Current best loss of DFNN: {best_loss:.6f}')

        self.model.load_state_dict(best_model)
        return best_loss
    
# Define Encoder 
# takes [B, n] return [B, N' = loop(floor((n + 2p - k) / s) + 1)**2)] 
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
# takes [B, N' = loop(floor((n + 2p - k) / s) + 1)**2)] return [B, n]
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
    def __init__(self, nt, DOD_DL_model, Coeff_DOD_DL_model, Encoder_model, Decoder_model, train_valid_set, error_weight=0.5,
                 epochs=1, restarts=1, learning_rate=1e-3, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.nt = nt
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
        """
        Dimensionality (this variant):
        V = innerDOD(mu,t):                 [B, N_A, N']   # Euclidean-orthonormal columns
        coeff_pred = Coeff_DOD_DL(...):     [B, n]         # DFNN output
        decoder(coeff_pred):                [B, N']        # expands n -> N'  (beta_pred)
        V @ beta_pred:                      [B, N_A]       # ambient recon from predicted coeffs (u_pred)
        encoder(alpha_vec in DOD-space):    [B, n]         # compress N' -> n
        solution_batch[:, i, :]:            [B, N_A]       # ambient POD solution at time i
        alpha_true = V^T u:                 [B, N']        # true N' coeffs from data
        """
        B = mu_batch.size(0)
        temp_error = 0.0

        for i in range(self.nt + 1):
            t_batch = torch.full((B, 1), i / (self.nt + 1),
                                dtype=torch.float32, device=self.device)                # [B,1]

            coeff_pred = self.coeff_model(mu_batch, nu_batch, t_batch)                  # [B, n]
            beta_pred  = self.de_model(coeff_pred)                                      # [B, N']

            V = self.innerDOD(mu_batch, t_batch)                                        # [B, N_A, N']
            u_pred = torch.bmm(V, beta_pred.unsqueeze(-1)).squeeze(-1)                  # [B, N_A]

            u = solution_batch[:, i, :]                                                 # [B, N_A]

            alpha_true = torch.bmm(V.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)      # [B, N']

            enc_of_target = self.en_model(alpha_true)                                   # [B, n]

            dynam_error = u - u_pred                                                    # [B, N_A]
            proj_error  = enc_of_target - coeff_pred                                    # [B, n]

            temp_error = temp_error + (
                self.error_weight * 0.5 * torch.sum(torch.norm(dynam_error, dim=1) ** 2) +
                (1.0 - self.error_weight) * 0.5 * torch.sum(torch.norm(proj_error,  dim=1) ** 2)
            )

        loss = temp_error / (B * (self.nt + 1))
        return loss



    def train(self):
        best_model = None
        best_loss = float('inf')
        best_loss_restart = float('inf')

        tqdm.write("Model DOD DL ROM is being trained...")

        for restart_idx in tqdm(range(self.restarts), desc="Restarts DOD DL ROM", leave=False):
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

            tqdm.write(f'Current best loss of DOD DL ROM: {best_loss:.6f}')

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
    def __init__(self, nt, Coeff_model, Encoder_model, Decoder_model, train_valid_set, error_weight,
                 epochs=1, restarts=1, learning_rate=1e-3, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.nt = nt
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
        for i in range(self.nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i / (self.nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            coeff_output = self.coeff_model(mu_batch, nu_batch, t_batch)           # [B, n]
            decoder_output = self.de_model(coeff_output)                           # [B, N_A]

            solution_slice = solution_batch[:, i, :]                               # [B, N_A]
            encoder_output = self.en_model(solution_slice)                         # [B, N]

            dynam_error = solution_slice - decoder_output                          # [B, N_A]
            proj_error = encoder_output - coeff_output                             # [B, n]
            temp_error += (self.error_weight / 2 * torch.sum(torch.norm(dynam_error, dim=1) ** 2) 
                           + (1-self.error_weight) / 2 * torch.sum(torch.norm(proj_error, dim=1) ** 2))

        loss = temp_error / (batch_size * (self.nt + 1))
        return loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        best_loss_restart = float('inf')

        tqdm.write("Model POD DL ROM is being trained...")


        for restart_idx in tqdm(range(self.restarts), desc="Restarts POD DL ROM", leave=False):
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

            tqdm.write(f'Current best loss of POD DL ROM: {best_loss:.6f}')

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
    def __init__(self, nt, DOD_0_model, coeffnn_0_model, colora_model, train_valid_set, epochs, restarts, learning_rate,
                 batch_size, device='cuda' if torch.cuda.is_available() else 'cpu', patience=3):
        self.nt = nt
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
        for i in range(self.nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i  / (self.nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            coeff_0_output = self.model_0(mu_batch, nu_batch).unsqueeze(2)
            DOD_0_output = self.DOD_0(mu_batch)                         

            output = self.model(torch.bmm(
                DOD_0_output.transpose(1, 2), coeff_0_output), nu_batch, t_batch)     # [B, N', N_A]

            u_proj = solution_batch[:, i, :]                                            

            error = output - u_proj                                                   # [B, N_A]
            temp_error += torch.sum(torch.norm(error, dim=1) ** 2)

        loss = temp_error / (batch_size * (self.nt + 1))
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
-------------------
Define simple forward pass assuming existence of networks
-------------------
'''

def denormalize_solution(sol_norm: torch.Tensor, example_name: str, reduction_tag: str):
    stats_path = f'examples/{example_name}/training_data/normalization_{reduction_tag}_{example_name}.npz'
    d = np.load(stats_path)
    m = torch.tensor(d['sol_min'], dtype=sol_norm.dtype, device=sol_norm.device)  # [D]
    M = torch.tensor(d['sol_max'], dtype=sol_norm.dtype, device=sol_norm.device)  # [D]

    scale = (M - m)

    if sol_norm.ndim == 3:   # [Ns, Nt, D]
        return sol_norm * scale.unsqueeze(0).unsqueeze(0) + m.unsqueeze(0).unsqueeze(0)
    elif sol_norm.ndim == 2: # [Nt, D] or [Ns, D]
        return sol_norm * scale.unsqueeze(0) + m.unsqueeze(0)
    elif sol_norm.ndim == 1: # [D]
        return sol_norm * scale + m
    else:
        raise ValueError(f"Unexpected ndim={sol_norm.ndim} for sol_norm")



def pod_dl_rom_forward(A_P, POD_DL_model, De_model, 
                       mu_i, nu_i, nt_, example_name: str, reduction_tag: str='N_reduced'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preds = []
    for j in range(nt_ + 1):
        t = torch.tensor(j / (nt_ + 1), dtype=torch.float32, device=device).view(1,1)
        coeff = POD_DL_model(mu_i, nu_i, t).unsqueeze(0)                 # [1, n]
        y_norm = De_model(coeff).squeeze(0)                              # [N] (normalized reduced)
        y = denormalize_solution(y_norm, example_name, reduction_tag)    # [N]
        u = torch.matmul(A_P, y)                                         # [N_A]
        preds.append(u)
    return torch.stack(preds, dim=0).detach().cpu().numpy()

def dod_dfnn_forward(A, innerDOD_model, DFNN_Nprime_model,
                     mu_i, nu_i, nt_, example_name: str, reduction_tag: str='N_A_reduced'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preds = []
    for j in range(nt_ + 1):
        t = torch.tensor(j / (nt_ + 1), dtype=torch.float32, device=device).view(1,1)
        V_tilde = innerDOD_model(mu_i, t)                                        # [N_A, N']
        coeff = DFNN_Nprime_model(mu_i, nu_i, t).squeeze(0)                         # [N']
        u_red_norm = torch.matmul(V_tilde, coeff)                                # [N_A]
        u_red = denormalize_solution(u_red_norm, example_name, reduction_tag)    # [N_A]
        u = torch.matmul(A, u_red)                                               # [Nh]
        preds.append(u)
    return torch.stack(preds, dim=0).detach().cpu().numpy()


def dod_dl_rom_forward(A, innerDOD_model, DOD_DL_model, De_model, 
                       mu_i, nu_i, nt_, example_name: str, reduction_tag: str='N_A_reduced'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preds = []
    for j in range(nt_ + 1):
        t = torch.tensor(j / (nt_ + 1), dtype=torch.float32, device=device).view(1,1)
        V_tilde = innerDOD_model(mu_i, t)                                      # [N_A, N']
        coeff_n = DOD_DL_model(mu_i, nu_i, t)                                  # [1, n]
        beta = De_model(coeff_n).squeeze(0)                                    # [N'] 
        u_red_norm = torch.matmul(V_tilde, beta)                               # [N_A] (normalized)
        u_red = denormalize_solution(u_red_norm, example_name, reduction_tag)  # [N_A]
        u = torch.matmul(A, u_red)                                             # [N_h]
        preds.append(u)
    return torch.stack(preds, dim=0).detach().cpu().numpy()

def colora_forward(A, stat_DOD_model, stat_Coeff_model, CoLoRA_DL_model, 
                   mu_i, nu_i, nt_, example_name: str, reduction_tag: str='N_A_reduced'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preds = []
    for j in range(nt_ + 1):
        t = torch.tensor(j / (nt_ + 1), dtype=torch.float32, device=device).view(1,1)
        coeff0 = stat_Coeff_model(mu_i, nu_i).unsqueeze(0).unsqueeze(2)   # [1, N', 1]
        V0 = stat_DOD_model(mu_i)                                         # [1, N_A, N']
        v_0 = torch.bmm(V0.transpose(1, 2), coeff0)                       # [N_A]
        u_norm = CoLoRA_DL_model(v_0, nu_i, t).squeeze(0)                 # [N_A] (normalized reduced)
        u = denormalize_solution(u_norm, example_name, reduction_tag)     # [N_A]
        u = torch.matmul(A, u)                                            # lift
        preds.append(u)
    return torch.stack(preds, dim=0).detach().cpu().numpy()