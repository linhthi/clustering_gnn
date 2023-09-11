import torch
import torch.nn as nn
import math
import dgl.function as fn
from  evaluation.evaluation import evaluation

def train_epoch(mode, optimizer, device, graph):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    epoch_train_f1 = 0
    epoch_train_nmi = 0
    epoch_train_ari = 0

    graph = graph.to(device)
    feats = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)

    optimizer.zero_grad()
    adj_rec = model.forward(graph)
    adj_orig = graph.adjacency_matrix().to_dense().to(device)
    pos_weight = ((adj_orig.shape[0] * adj_orig.shape[0]) - adj_orig.sum()) / adj_orig.sum()

    loss = model.loss(adj_rec, adj_orig, pos_weight)
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()

    epoch_train_acc, epoch_train_f1, epoch_train_nmi, epoch_train_ari = evaluation(labels, adj_rec.detach().cpu())
    evals = [epoch_train_acc, epoch_train_f1, epoch_train_nmi, epoch_train_ari]

    return epoch_loss, evals

def evaluate_network(model, device, graph):
    model.eval()
    epoch_test_acc = 0
    epoch_test_f1 = 0
    epoch_test_nmi = 0
    epoch_test_ari = 0

    graph = graph.to(device)
    feats = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)

    adj_rec = model.forward(graph)
    adj_orig = graph.adjacency_matrix().to_dense().to(device)
    pos_weight = ((adj_orig.shape[0] * adj_orig.shape[0]) - adj_orig.sum()) / adj_orig.sum()
    loss = model.loss(adj_rec, adj_orig, pos_weight)

    epoch_test_loss = loss.detach().item()
    epoch_test_acc, epoch_test_f1, epoch_test_nmi, epoch_test_ari = evaluation(labels, adj_rec.detach().cpu())
    evals = [epoch_test_acc, epoch_test_f1, epoch_test_nmi, epoch_test_ari]

    return epoch_test_loss, evals
    