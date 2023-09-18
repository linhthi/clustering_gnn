import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as function
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(g, h)
        return h

class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def forward(self, x, **kwargs):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        return x

