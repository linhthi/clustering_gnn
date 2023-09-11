import random
import numpy as np
from evaluation.evaluation import evaluation
from sklearn.cluster import KMeans

def k_means(embedding, k, y_true, device):
    """
    K-means clustering.
    :param embedding: embedding matrix
    :param k: number of clusters
    :param y_true: ground truth labels
    :param device: which device to use
    :return: clustering results
    """

    if device == "cpu":
        model = KMeans(n_clusters=k, n_init=20)
        cluster_id = model.fit_predict(embedding)
        center = model.cluster_centers_
    if device == 'gpu':
        pass
    acc, f1, nmi, ari = evaluation(y_true, cluster_id)
    return acc, f1, nmi, ari, cluster_id, center