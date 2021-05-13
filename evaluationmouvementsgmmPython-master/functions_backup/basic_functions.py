import numpy as np


def acoslog(x):
    '''
    Arcosine re-defitinion to make sure the distance between antipodal quaternions is zero (2.50 from Dubbelman's Thesis)
    '''
    x = np.maximum(np.minimum(x, 1), -1)
    return np.where(x < 0, np.arccos(x) - np.pi, np.arccos(x))


def QuatMatrix(q):
    q = np.reshape(q,4)  # make sure q is a 1-d vector
    return np.array([[q[0], -q[1], -q[2], -q[3]],
                         [q[1], q[0], -q[3], q[2]],
                         [q[2], q[3], q[0], -q[1]],
                         [q[3], -q[2], q[1], q[0]]])

def round(x):
    '''
    :param x: must be a positive array
    '''
    ##TODO ： if x not positive, raise error
    xlfoor = np.floor(x)
    return np.where(x - xlfoor < 0.5, xlfoor, xlfoor + 1)