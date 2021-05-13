from functions.mapping import logmap
from functions.dp import *


def temporalAlignment(trainD, testD, fast, sizeData):
    # Dynamic programming to align two sequences train and test
    # fast may be set to 1 to speed process by using small windows of 70
    # Compute distances
    DM = np.zeros((sizeData, sizeData))
    for row in range(0, sizeData):  ##TODO : possible optimisation
        DMline = logmap(testD['lElbow ori'], trainD['lElbow ori'][:, row:row + 1])
        DMline = np.vstack((DMline, logmap(testD['lWrist ori'], trainD['lWrist ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['lShoulder ori'], trainD['lShoulder ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['rElbow ori'], trainD['rElbow ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['rWrist ori'], trainD['rWrist ori'][:, row:row + 1])))
        DMline = np.vstack((DMline, logmap(testD['rShoulder ori'], trainD['rShoulder ori'][:, row:row + 1])))
        DM[row, :] = np.linalg.norm(DMline, axis=0)
    if fast == 1:
        for row in range(sizeData - 70):
            for col in range(70 + row, sizeData):
                DM[row][col] = 5
        for col in range(sizeData - 70):
            for row in range(70 + col, sizeData):
                DM[row][col] = 5
    p, q = dp(DM)  ## execution time : 1s
    # alignment
    r = np.zeros(sizeData)
    for t in range(sizeData):
        ind = np.where(p == t)[0]
        if ind.size == 0:
            r[t] = p[t]
        else:
            r[t] = q[ind[0]]
    # out
    r = r.astype(int)
    out = {}
    for key in testD:
        out[key] = testD[key][:, r]
    return out, r
