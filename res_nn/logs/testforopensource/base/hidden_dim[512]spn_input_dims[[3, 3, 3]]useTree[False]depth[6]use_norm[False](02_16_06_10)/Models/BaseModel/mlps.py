import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, depth, dropout, use_norm):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([])
        for i in range(depth-1):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            if use_norm:
                self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, use_norm):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.norm = None
        if use_norm:
            self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        short_cut = x
        if self.norm is not None:
            x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        x = short_cut + x
        return x

class ResMLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, depth, dropout, use_norm):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([])

        for i in range(depth-1):
            if i==0:
                self.layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.layers.append(ResBlock(hidden_dim, hidden_dim, dropout, use_norm))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x