import ot
import numpy as np
from scipy.spatial import distance_matrix

def compute_map(A, B, metric="euclidean"):
    M = compute_dist(A,B)

    na = A.shape[0]
    a = np.ones(na) / na

    nb = B.shape[0]
    b = np.ones(nb) / nb

    return ot.emd(a, b, M)


def compute_wasserstein(A, B, metric="euclidean"):
    M = compute_dist(A,B)

    na = A.shape[0]
    a = np.ones(na) / na

    nb = B.shape[0]
    b = np.ones(nb) / nb

    return ot.emd2(a, b, M)

def compute_dist(A,B, metric="euclidean"):
    return ot.dist(A, B, metric)

def compute_entropic(A,B, reg=1, metric="euclidean"):
    M = compute_dist(A,B)

    na = A.shape[0]
    a = np.ones(na) / na

    nb = B.shape[0]
    b = np.ones(nb) / nb

    return ot.sinkhorn(a, b, M, reg)

def get_matches(index, mapping):
    """

    :param index: the index of the row you want
    :param mapping: the transport mapping
    :return: return the indicies of the mapped objects
    """
    #get the row with the specified index
    index_row = mapping[index]

    #get the indices where there is a non-zero transport mass
    indices = np.where(index_row != 0)

    return indices


def unfairness_metric(T, X0, X1):
    d = distance_matrix(X0,X1)
    uf = np.sum( np.multiply(T, d), axis = 1) # element-wise mult + row sums
    return uf

def unfairness_metric_norm(T, X0, X1):
    """

    :param T: The Transport map/optimal coupling
    :param X: Features 
    :return:
    """
    d = distance_matrix(X0,X1)
    max_distance = np.amax(d, axis = 1) # get max along rows
    uf = np.sum( np.multiply(T, d), axis = 1) # element-wise mult + row sums
    uf_norm = np.divide(uf, max_distance) * X0.shape[0] # normalize by worst possible case

    return uf_norm
