import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Some Constants
time_end = 1.
nt = 10

# Initialize linear weights
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

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

# Define Seed Module for geometric parameter
class SeedModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SeedModule, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, mu):
        mu = func.leaky_relu(self.fc(mu), 0.1)
        return mu

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
        t = t.unsqueeze(-1)  # Now shape becomes (1,) if t was a scalar
        input_tensor = torch.cat((nu, t.expand(nu.shape[0], 1)), dim=-1)
        out = self.fc(input_tensor)
        return out


# Define Root Module with time-continuity
class RootModule(nn.Module):
    def __init__(self, N_h, L, seed_dim, rank, physical_dim, with_bias=True, param_dtype=torch.float32):
        super().__init__()
        self.N_h = N_h
        self.L = L
        self.width = seed_dim
        self.rank = rank
        self.with_bias = with_bias
        self.param_dtype = param_dtype

        self.w_init = nn.init.kaiming_normal_
        self.b_init = nn.init.zeros_
        self.z_init = nn.init.zeros_

        self.Ws = nn.ParameterList([
            nn.Parameter(torch.empty((self.N_h, self.width), dtype=self.param_dtype))
            for _ in range(L)
        ])
        self.As = nn.ParameterList([
            nn.Parameter(torch.empty((self.N_h, self.rank), dtype=self.param_dtype))
            for _ in range(L)
        ])
        self.Bs = nn.ParameterList([
            nn.Parameter(torch.empty((self.rank, self.width), dtype=self.param_dtype))
            for _ in range(L)
        ])

        self.alphas = nn.ModuleList([Alpha(physical_dim) for _ in range(L)])

        if self.with_bias:
            self.bs = nn.ParameterList([
                nn.Parameter(torch.zeros(self.N_h, dtype=self.param_dtype))
                for _ in range(L)
            ])

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.L):
            self.w_init(self.Ws[i])
            self.w_init(self.As[i])
            self.z_init(self.Bs[i])
            if self.with_bias:
                self.b_init(self.bs[i])

    def forward(self, mu, nu, t):
        # Use a temporary variable x for iterative updates.
        x = mu  # x should have shape (B, width) where width == N_h
        for i in range(self.L):
            alpha_val = self.alphas[i](nu, t).unsqueeze(1)  # (B, 1, 1)
            A_exp = self.As[i].unsqueeze(0)  # (1, N_h, rank)
            AB = (A_exp * alpha_val) @ self.Bs[i]  # (B, N_h, width)
            W = self.Ws[i] + AB  # (B, N_h, width)
            x_unsq = x.unsqueeze(2)  # (B, width, 1) -> here width should equal N_h
            x = torch.bmm(W, x_unsq).squeeze(2)  # (B, N_h, 1) -> (B, N_h)
            if self.with_bias:
                x = x + self.bs[i].unsqueeze(0).expand_as(x)
        return x


# Define Complete DOD DL Model (returns a tensor of size [N_h, n])
class DOD_DL(nn.Module):
    def __init__(self, N_h, L, seed_dim, geometric_dim, rank, physical_dim, n):
        super(DOD_DL, self).__init__()
        self.seed_module = SeedModule(geometric_dim, seed_dim)
        self.root_modules = nn.ModuleList([RootModule(N_h, L, seed_dim,
                                                      rank, physical_dim) for _ in range(n)])

    def forward(self, mu, nu, t):
        seed_output = self.seed_module(mu)
        root_outputs = [root(seed_output, nu, t) for root in self.root_modules]
        v_mu = torch.stack(root_outputs, dim=0).transpose(0,1)
        return v_mu

# Define DOD-DL training
class DOD_DL_Trainer:
    def __init__(self, nn_model, train_valid_set, epochs=1, restart=1, learning_rate=1e-3,
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

    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)  # Get the batch size (can be problem, if set is non-divisible)
        temp_loss = 0.

        for i in range(nt + 1):
            output = self.model(mu_batch, nu_batch, i/(nt + 1))

            # Perform batch matrix multiplications
            v_u = torch.bmm(output, solution_batch[:, i, :].unsqueeze(2))
            u_proj = torch.bmm(output.transpose(1, 2), v_u)

            error = solution_batch[:, i, :].unsqueeze(2) - u_proj
            temp_loss += torch.sum(torch.norm(error, dim=1) ** 2)

        loss = temp_loss / batch_size

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

                print(f"At Epoch {epoch} the loss is {total_loss / len(self.train_loader)}")

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for mu_batch, nu_batch, solution_batch in self.valid_loader:
                    val_loss += self.loss_function(mu_batch, nu_batch, solution_batch).item()
                val_loss /= len(self.valid_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()

            print(f'Restart DOD. Gen Count at {_ + 1} with current best loss {best_loss}')

        # Load the best model
        self.model.load_state_dict(best_model)
        return best_loss

# Define Phi_1 module for allocation
class Phi1Module(nn.Module):
    def __init__(self, geometric_dim, m_0, n_0, layer_sizes, leaky_relu_slope=0.1):
        super(Phi1Module, self).__init__()
        self.m = m_0
        self.n = n_0
        if layer_sizes is None:
            layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(geometric_dim, layer_sizes[0]))
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

# Define Phi_2 module for allocation
class Phi2Module(nn.Module):
    def __init__(self, physical_dim, m_0, n_0, layer_sizes, leaky_relu_slope=0.1):
        super(Phi2Module, self).__init__()
        self.m = m_0
        self.n = n_0
        if layer_sizes is None:
            layer_sizes = [1]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(physical_dim, layer_sizes[0]))
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

# Define Complete parameter-to-DOD-coefficients Model (returns a vector of size [n])
class Coeff_DOD_DL(nn.Module):
    def __init__(self, geometric_dim, physical_dim, m_0, n_0, layer_sizes=None):
        super(Coeff_DOD_DL, self).__init__()
        self.phi_1_module = Phi1Module(geometric_dim, m_0, n_0, layer_sizes)
        self.phi_2_module = Phi2Module(physical_dim, m_0, n_0, layer_sizes)

    def forward(self, mu, nu):
        phi_1 = self.phi_1_module(mu)
        phi_2 = self.phi_2_module(nu)
        phi = phi_1 * phi_2
        phi_sum = torch.sum(phi, dim=1).squeeze()
        return phi_sum


# Define the Trainer
class Coeff_DOD_DL_Trainer:
    def __init__(self, dod_model, coeffnn_model, train_valid_set, epochs=1, restarts=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.dod = dod_model.to(device)
        self.model = coeffnn_model.to(device)
        self.device = device

        self.G = torch.tensor(np.load('gram_matrix.npy', allow_pickle=True), dtype=torch.float32).to(self.device)

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)
        G_expanded = self.G.unsqueeze(0).expand(batch_size, -1, -1)

        temp_error = 0.0
        for i in range(nt + 1):
            # Get the coefficient model output; expected shape: (B, n)
            output = self.model(mu_batch, nu_batch)
            # Get the DOD model output; expected shape: (B, n, N_h)
            dod_output = self.dod(mu_batch, nu_batch, i / (nt + 1))

            # Extract the solution slice at time step i; expected shape: (B, N_h, 1)
            solution_slice = solution_batch[:, i, :].unsqueeze(2)

            # Compute u_proj:
            #   First: torch.bmm(G_expanded, solution_slice) has shape (B, N_h, 1)
            #   Then: torch.bmm(dod_output, ...) has shape (B, n, 1)
            u_proj = torch.bmm(dod_output, torch.bmm(G_expanded, solution_slice)).squeeze(2)  # now (B, n)

            error = output - u_proj  # should be (B, n)
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

                print(f"At Epoch {epoch} the loss is {total_loss / len(self.train_loader)}")

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
