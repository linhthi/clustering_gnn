import torch.nn as nn
import layers.encoder as enc
import layers.decoder as dec
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss

class GAE(nn.Module):
    """
    Graph Autoencoder (GAE)
    """
    
    def __init__(self, net_params):
        super(GAE, self).__init__()

        in_dim = net_params['in_dim']
        n_clusters = net_params['n_clusters']
        dropout = net_params['dropout']
        activation = net_params['activation']
        device = net_params['device']

        self.in_dim = in_dim
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.activation = activation
        self.device = device
        
        self.encoder = enc.GCN(in_dim, n_clusters, dropout, activation)
        self.decoder = dec.InnerProductDecoder(activation)

    def forward(self, g):
        h = self.encoder(g, g.ndata['feat'])
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def loss(adj_rec, adj_orig, pos_weight):
        loss = BCELoss(adj_rec, adj_orig, pos_weight)
        return loss