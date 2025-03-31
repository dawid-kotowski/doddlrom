import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Some Constants
time_end = 1.

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
        input_tensor = torch.cat((nu, t.unsqueeze(-1)), dim=-1)
        out = self.fc(input_tensor)
        return out


# Define Root Module with time-continuity
class RootModule(nn.Module):
    def __init__(self, N_h, L, geometric_dim, rank, physical_dim, with_bias=True, param_dtype=torch.float32):
        super().__init__()
        self.N_h = N_h
        self.L = L
        self.width = geometric_dim
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
        for i in range(self.L):
            alpha_val = self.alphas[i](nu, t).unsqueeze(1)  # now shape (B, 1, 1)
            A_exp = self.As[i].unsqueeze(0)  # now shape (1, N_h, rank)
            AB = (A_exp * alpha_val) @ self.Bs[i]  # result shape: (B, N_h, width)
            W = self.Ws[i] + AB
            mu = mu.unsqueeze(2)  # shape: (B, width, 1)
            mu = torch.bmm(W, mu)  # shape: (B, N_h, 1)
            mu = mu.squeeze(2)  # shape: (B, N_h)

            if self.with_bias:
                mu += self.bs[i].unsqueeze(0).expand_as(mu)

        return mu

# Define Complete DOD DL Model (returns a tensor of size [N_h, n])
class DOD_DL(nn.Module):
    def __init__(self, N_h, L, seed_dim, geometric_dim, rank, physical_dim, n):
        super(DOD_DL, self).__init__()
        self.seed_module = SeedModule(1, seed_dim)
        self.root_modules = nn.ModuleList([RootModule(N_h, L, geometric_dim,
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
        self.T = time_end
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

        for i in range(101):
            output = self.model(mu_batch, nu_batch, i/self.T)

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

