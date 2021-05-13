import numpy as np


def dp(M):
    """
    Use dynamic programming to find a min-cost path through matrix M.
    :return:  state sequence in p,q
    """
    r, c = M.shape

    # costs
    D = np.zeros((r + 1, c + 1))
    D[0:1, :] = np.inf
    D[:, 0:1] = np.inf
    D[0, 0] = 0
    D[1:r + 1, 1:c + 1] = M

    # traceback
    phi = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            triple = np.array([D[i, j], D[i, j + 1], D[i + 1, j]])
            tb = np.argmin(triple)
            D[i + 1, j + 1] += triple[tb]
            phi[i, j] = tb+1

    # Traceback from top left
    i = r-1
    j = c-1
    p = i
    q = j
    while i > 0 and j > 0:
        tb = phi[i, j]
        if tb == 1:
            i -= 1
            j -= 1
        elif tb == 2:
            i -= 1
        elif tb == 3:
            j -= 1
        else:
            raise Exception('error in dp')
        p = np.hstack((i, p))
        q = np.hstack((j, q))
    return p, q
