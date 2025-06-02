import numpy as np
import numba
from sklearn.preprocessing import normalize


def cos_dist(t1, t2):
    sum = np.sum(np.dot(t1, t2))
    return np.sqrt(sum)

def calculate_adj(patches):
    # patches: B × h × w
    # return: adj
    patches_flatten = patches.reshape([-1, 256*256])

    n = patches_flatten.shape[0]
    adj = np.empty((n,n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = cos_dist(patches_flatten[i], patches_flatten[j])
    return normalize(adj)