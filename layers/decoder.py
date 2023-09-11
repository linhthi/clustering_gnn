import torch
import torch.nn as nn
import torch.nn.functional as F

class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.activation(torch.mm(z, z.t()))
        return adj