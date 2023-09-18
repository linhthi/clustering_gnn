import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari
import torch

def evaluation(y_true, y_pred):
    """
    Evaluate clustering performance.
    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :return: ACC, F1, NMI and ARI
    - ACC: Accuracy
    - F1: F1 score
    - NMI: Normalized Mutual Information
    - ARI: Adjusted Rand Index
    """
    
    nmi_score = nmi(y_true, y_pred)
    ari_score = ari(y_true, y_pred)

    # y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_Class1 = len(l1)
    l2 = list(set(y_pred))
    num_Class2 = len(l2)
    ind = 0
    if num_Class1 != num_Class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_Class2 = len(l2)
    if num_Class1 != num_Class2:
        print('Class Not equal, Error!!!!!!!!!!!!!!!')
        return
    
    cost = np.zeros((num_Class1, num_Class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')

    return acc, f1, nmi_score, ari_score

# https://github.com/JuliaSun623/VGAE_dgl/blob/main/train.py#L38
def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0]) - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm