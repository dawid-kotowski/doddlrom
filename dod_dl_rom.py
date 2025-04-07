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

# Define DatasetLoader for DOD_DL
class DOD_DL_DatasetLoader(Dataset):
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
        u_new = u @ self.G @ self.A
        return mu, u_new



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

# Define Complete DOD_DL_ DL Model (returns a tensor of size [N_A, n])
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
    def __init__(self, nn_model, train_valid_set, N_A, epochs=1, restart=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.restarts = restart
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = nn_model.to(device)
        self.device = device
        self.A = torch.tensor(np.load('ambient_matrix.npy', allow_pickle=True), dtype=torch.float32).to(self.device)
        self.G = torch.tensor(np.load('gram_matrix.npy', allow_pickle=True), dtype=torch.float32).to(self.device)


        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DOD_DL_DatasetLoader(train_data, self.G, self.A, N_A), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DOD_DL_DatasetLoader(valid_data, self.G, self.A, N_A), batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, solution_batch):
        batch_size = mu_batch.size(0)  # Get the batch size (can be problem, if set is non-divisible)
        temp_loss = 0.

        for i in range(nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i/(nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
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
                for mu_batch, solution_batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = self.loss_function(mu_batch, solution_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"At Epoch {epoch} the loss is {total_loss / len(self.train_loader)}")

            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for mu_batch, solution_batch in self.valid_loader:
                    val_loss += self.loss_function(mu_batch, solution_batch).item()
                val_loss /= len(self.valid_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model.state_dict()

            print(f'Restart DOD_DL_. Gen Count at {_ + 1} with current best loss {best_loss}')

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

# Define Complete parameter-to-DOD_DL_-coefficients Model (returns a vector of size [n])
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
    def __init__(self, DOD_DL_model, coeffnn_model, train_valid_set, epochs=1, restarts=1, learning_rate=1e-3,
                 batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.restarts = restarts
        self.batch_size = batch_size
        self.DOD_DL = DOD_DL_model.to(device)
        self.model = coeffnn_model.to(device)
        self.device = device

        self.G = torch.tensor(np.load('gram_matrix.npy', allow_pickle=True), dtype=torch.float32).to(self.device)
        self.A = torch.tensor(np.load('ambient_matrix.npy', allow_pickle=True), dtype=torch.float32).to(self.device)

        train_data = train_valid_set('train')
        valid_data = train_valid_set('valid')

        self.train_loader = DataLoader(DatasetLoader(train_data), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(DatasetLoader(valid_data), batch_size=self.batch_size, shuffle=False)

    def loss_function(self, mu_batch, nu_batch, solution_batch):
        batch_size = mu_batch.size(0)
        G_red = self.A.t() @ self.G @ self.A
        G_expanded = G_red.unsqueeze(0).expand(batch_size, -1, -1)

        temp_error = 0.0
        for i in range(nt + 1):
            t_batch = torch.stack(
                [torch.tensor(i / (nt + 1), dtype=torch.float32, device=self.device) for _ in range(batch_size)]
            ).unsqueeze(1)
            # Get the coefficient model output; expected shape: (B, n)
            output = self.model(mu_batch, nu_batch, t_batch)
            # Get the DOD_DL_ model output; expected shape: (B, n, N_h)
            DOD_DL_output = self.DOD_DL(mu_batch, t_batch)

            # Extract the solution slice at time step i; expected shape: (B, N_h)
            solution_slice = solution_batch[:, i, :]
            solution_reduced = solution_slice @ self.A
            solution_reduced = solution_reduced.unsqueeze(2) # shape: (B, N_A, 1)

            # Now perform the batch matrix multiplications:
            # inner: (B, N_A, 1)
            inner = torch.bmm(G_expanded, solution_reduced)
            # u_proj: (B, n, 1) then squeezed to (B, n)
            u_proj = torch.bmm(DOD_DL_output, inner).squeeze(2)

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
