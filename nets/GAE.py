import torch.nn as nn
import layers.encoder as enc
import layers.decoder as dec
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
import torch
from layers.clustering import k_means

class GAE(nn.Module):
    """
    Graph Autoencoder (GAE)
    """

    def __init__(self, net_params):
        super(GAE, self).__init__()

        in_dim = net_params['in_dim']
        n_clusters = net_params['n_classes']
        hidden_dim = net_params['hidden_dim']
        n_layers = net_params['n_layers']
        dropout = net_params['dropout']
        activation = torch.nn.functional.relu
        device = net_params['device']

        self.in_dim = in_dim
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.activation = activation
        self.device = device
        
        self.encoder = enc.GCN(in_dim, hidden_dim, n_clusters, n_layers, activation, dropout)
        self.decoder = dec.InnerProductDecoder(activation)

    def forward(self, g):
        h = self.encoder(g, g.ndata['feat'])
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        y_pred, center = k_means(h.detach().cpu().numpy(), self.n_clusters)
        return adj_rec, y_pred, center

    def loss(self, adj_rec, adj_orig):
        loss = BCELoss(adj_rec, adj_orig)
        return loss