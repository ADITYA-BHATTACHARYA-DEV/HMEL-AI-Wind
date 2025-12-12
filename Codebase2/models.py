# File 2: models.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# -----------------------------
# PINN Model
# -----------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# BiLSTM Model
# -----------------------------
class BiLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -----------------------------
# Graph Neural Network Model
# -----------------------------
class WindGNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
