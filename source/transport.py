import ot
import numpy as np

def compute_map(A, B, metric="euclidean"):
    M = ot.dist(A, B, metric)

    na = A.shape[0]
    a = np.ones(na) / na

    nb = B.shape[0]
    b = np.ones(nb) / nb

    return ot.emd(a, b, M)


def compute_wasserstein(A, B, metric="euclidean"):
    M = ot.dist(A, B, metric)

    na = A.shape[0]
    a = np.ones(na) / na

    nb = B.shape[0]
    b = np.ones(nb) / nb

    return ot.emd2(a, b, M)


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



