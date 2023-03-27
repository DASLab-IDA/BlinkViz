import torch
import torch.nn as nn
import torch.nn.functional as F
from . import tcnn

class TreeConvolution(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.layers.append(tcnn.BinaryTreeConv(in_dim, hidden_dim))
        self.layers.append(tcnn.TreeLayerNorm())
        self.layers.append(tcnn.TreeActivation(nn.LeakyReLU()))
        self.layers.append(tcnn.BinaryTreeConv(hidden_dim, int(hidden_dim/2)))
        self.layers.append(tcnn.TreeLayerNorm())
        self.layers.append(tcnn.TreeActivation(nn.LeakyReLU()))
        self.layers.append(tcnn.BinaryTreeConv(int(hidden_dim/2), out_dim))
        self.layers.append(tcnn.TreeLayerNorm())
        self.layers.append(tcnn.TreeActivation(nn.LeakyReLU()))
        self.layers.append(tcnn.DynamicPooling())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class network(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg # config

        if cfg.res_mlp:
            from .mlps import ResMLP as MLP
            print("[Using ResMLP]")
        else:
            from .mlps import MLP

        self.spn_input_dims = cfg.spn_input_dims

        self.spn_num = len(self.spn_input_dims)
        if cfg.useTree:
            self.spn_num = int(self.spn_num/2)

        if cfg.useTree:
            self.spn_encoders =  nn.ModuleList([TreeConvolution(2, cfg.hidden_dim, cfg.hidden_dim//4) for idx in range(0, self.spn_num)])
        else:
            self.spn_encoders = nn.ModuleList([MLP(self.spn_input_dims[idx]+1, cfg.hidden_dim, cfg.hidden_dim, cfg.depth, 0, use_norm=cfg.use_norm) for idx in range(self.spn_num)])
        self.preds_encoder = MLP(self.spn_num, cfg.hidden_dim//2, cfg.hidden_dim//2, cfg.depth, 0, use_norm=cfg.use_norm)
        self.fusion_layer = MLP(self.spn_num*cfg.hidden_dim+cfg.hidden_dim//2+1, cfg.hidden_dim, cfg.hidden_dim, cfg.depth, 0, use_norm=cfg.use_norm)
        self.decoder = MLP(cfg.hidden_dim+1, cfg.hidden_dim, 1, cfg.depth, 0, use_norm=cfg.use_norm)

    def forward(self, spn_values, spn_preds, output):
        if self.cfg.zero_debug:
            spn_values = spn_values * 0
        spn_values = torch.split(spn_values, dim=1, split_size_or_sections=self.spn_input_dims)
        if self.cfg.useTree:
            spn_tree_values = []
            for idx in range(len(self.spn_input_dims)):
                if idx%2==0:
                    spn_tree_values.append(spn_values[idx].reshape(spn_values[idx].shape[0], int(self.spn_input_dims[idx]/2), 2))
                else:
                    spn_tree_values.append(spn_values[idx].reshape(spn_values[idx].shape[0], self.spn_input_dims[idx], 1))
            spn_values = spn_tree_values
        spn_preds = torch.split(spn_preds, dim=1, split_size_or_sections=[1]*self.spn_num)

        encoded_features = []
        if self.cfg.useTree:
            for idx in range(0, self.spn_num):
                encoded_features.append(self.spn_encoders[idx]((spn_values[idx*2].transpose(1,2),spn_values[idx*2+1].to(torch.int64))))
            encoded_features = torch.cat(encoded_features, dim=1)
        else:
            for idx in range(self.spn_num):
                encoded_features.append(self.spn_encoders[idx](torch.cat([spn_values[idx], spn_preds[idx]], dim=1)))
            encoded_features = torch.cat(encoded_features, dim=1)

        spn_preds = torch.cat(spn_preds, dim=1)
        
        spn_preds_mean = torch.mean(spn_preds, dim=1, keepdim=True)
        preds_encoded_feature = self.preds_encoder(spn_preds)
        fused_feature = self.fusion_layer(torch.cat([encoded_features, preds_encoded_feature, spn_preds_mean], dim=1))
        res = None
        if self.cfg.useMul:
            res = spn_preds_mean * self.decoder(torch.cat([fused_feature, spn_preds_mean], dim=1)) + spn_preds_mean
        else:
            res = spn_preds_mean + self.decoder(torch.cat([fused_feature, spn_preds_mean], dim=1))
        
        return res
