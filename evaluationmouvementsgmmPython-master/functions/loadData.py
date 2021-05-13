import numpy as np
from functions.estimateOrientationFromPosition import estimateOrientationFromPosition
from functions.removeStart import removeStart
import scipy.signal
from scipy.interpolate import splev, splrep


##this script has been verified

def loadData(dir, fname, filt, est, rem, ws, resampleSize):
    path = dir + fname

    # load data
    with open(path, 'r+', encoding='utf-8') as f:
        M = np.array([i[:-2].split(' ') for i in f.readlines()]).astype(
            float)  ##if we use i[:-1], there will be a '' at the end of each line
    #oriMat = M[:, 3:7]
    #posMat = M[:, 0:3]
    posMat=M
    for j in range(1, 25):
        fO = j * 7 + 3
        #oriMat = np.hstack((oriMat, M[:, fO:fO + 4]))
        fP = j * 7
        posMat = np.hstack((posMat, M[:, fP:fP + 3]))

    # Filtering
    if filt == 1:
        b, a = scipy.signal.butter(3, 0.05)
        posMat = scipy.signal.lfilter(b, a, posMat, axis=0)
        #oriMat = scipy.signal.filtfilt(b, a, oriMat, axis=0)

    # Estimation
    if est == 1:
        oriMat = estimateOrientationFromPosition(posMat)

    # data structure : dictionnary
    data = {'lElbow ori': oriMat[:, 20:24].T,
            'lWrist ori': oriMat[:, 24:28].T,
            'lShoulder ori': oriMat[:, 16:20].T,
            'rElbow ori': oriMat[:, 36:40].T,
            'rWrist ori': oriMat[:, 40:44].T,
            'rShoulder ori': oriMat[:, 32:36].T,
            'mSpine ori': oriMat[:, 4:8].T,
            'mShoulder ori': oriMat[:, 80:84].T,
            'Neck ori': oriMat[:, 8:12].T,
            'lElbow rel_pos': (posMat[:, 15:18] - posMat[:, 6:9]).T,
            'lWrist rel_pos': (posMat[:, 18:21] - posMat[:, 6:9]).T,
            'lShoulder rel_pos': (posMat[:, 12:15] - posMat[:, 6:9]).T,
            'rElbow rel_pos': (posMat[:, 27:30] - posMat[:, 6:9]).T,
            'rWrist rel_pos': (posMat[:, 30:33] - posMat[:, 6:9]).T,
            'rShoulder rel_pos': (posMat[:, 24:27] - posMat[:, 6:9]).T,
            }
    oriMat = oriMat.T
    posMat = posMat.T

    if rem == 1:
        _in = data['lElbow ori']
        _in = np.vstack((_in, data['lWrist ori']))
        _in = np.vstack((_in, data['lShoulder ori']))
        _in = np.vstack((_in, data['rElbow ori']))
        _in = np.vstack((_in, data['rWrist ori']))
        _in = np.vstack((_in, data['rShoulder ori']))
        (out, deb) = removeStart(_in, ws, 0.001, 10)

        oriMat = oriMat[:, deb - 1:]
        posMat = posMat[:, deb - 1:]
        for item in data:
            data[item] = data[item][:, deb - 1:]

    for item in data:
        length = data[item].shape[1]
        new_x = np.linspace(1, length, resampleSize)
        data[item] = np.array([splev(new_x, splrep(range(1, length + 1), line, k=3)) for line in data[item]])

    length = oriMat.shape[1]
    new_x = np.linspace(1, length, resampleSize)
    oriMat = np.array([splev(new_x, splrep(range(1, length + 1), line, k=3)) for line in oriMat])

    length = posMat.shape[1]
    new_x = np.linspace(1, length, resampleSize)
    posMat = np.array([splev(new_x, splrep(range(1, length + 1), line, k=3)) for line in posMat])

    return oriMat, posMat, data
