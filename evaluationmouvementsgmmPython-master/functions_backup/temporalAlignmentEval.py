from functions.mapping import logmap
from functions.dp import *
from functions.basic_functions import round


def temporalAlignmentEval(m_cuts, trainD, testD, fast, sizeData):
    keypose = 0
    prevDist = 5
    poseTracking = np.zeros(m_cuts.size)
    motionStarted = 0
    performedEntirely = 0
    if fast == 1:
        resampleSize = 10
    else:
        resampleSize = 1

    row = np.arange(0, sizeData + 1, resampleSize).astype(int)
    col = np.arange(0, sizeData + 1, resampleSize).astype(int)
    DM = np.zeros((row.size, col.size))
    if row[-1] == sizeData:
        row[-1] = sizeData - 1
    if col[-1] == sizeData:
        col[-1] = sizeData - 1
    for irow in row:  ##TODO : possible optimisation
        Line = logmap(testD['lElbow ori'][:,col], trainD['lElbow ori'][:, irow:irow + 1])
        Line = np.vstack((Line, logmap(testD['lWrist ori'][:,col], trainD['lWrist ori'][:, irow:irow + 1])))
        Line = np.vstack((Line, logmap(testD['lShoulder ori'][:,col], trainD['lShoulder ori'][:, irow:irow + 1])))
        Line = np.vstack((Line, logmap(testD['rElbow ori'][:,col], trainD['rElbow ori'][:, irow:irow + 1])))
        Line = np.vstack((Line, logmap(testD['rWrist ori'][:,col], trainD['rWrist ori'][:, irow:irow + 1])))
        Line = np.vstack((Line, logmap(testD['rShoulder ori'][:,col], trainD['rShoulder ori'][:, irow:irow + 1])))
        DM[np.where(row == irow), :] = np.linalg.norm(Line, axis=0)
    if fast == 1:
        for i in range(15):
            for j in range(15 + i, 31):
                DM[i][j] = 0
        for j in range(15):
            for i in range(15 + j, 31):
                DM[i][j] = 0

    ## keyposeTracking
    i = m_cuts[keypose]-1
    Line = logmap(testD['lElbow ori'][:, col], trainD['lElbow ori'][:, i:i + 1])
    Line = np.vstack((Line, logmap(testD['lWrist ori'][:, col], trainD['lWrist ori'][:, i:i + 1])))
    Line = np.vstack((Line, logmap(testD['lShoulder ori'][:, col], trainD['lShoulder ori'][:, i:i + 1])))
    Line = np.vstack((Line, logmap(testD['rElbow ori'][:, col], trainD['rElbow ori'][:, i:i + 1])))
    Line = np.vstack((Line, logmap(testD['rWrist ori'][:, col], trainD['rWrist ori'][:, i:i + 1])))
    Line = np.vstack((Line, logmap(testD['rShoulder ori'][:, col], trainD['rShoulder ori'][:, i:i + 1])))
    dist = np.linalg.norm(Line, axis=0)
    for n in range(col.size):
        if performedEntirely == 0:
            if prevDist < 1.7 and dist[n] > prevDist:
                poseTracking[keypose] = 1
                keypose += 1
                if keypose == m_cuts.size:
                    performedEntirely = 1
                else:
                    i = m_cuts[keypose]-1
                    Line = logmap(testD['lElbow ori'][:, col], trainD['lElbow ori'][:, i:i + 1])
                    Line = np.vstack((Line, logmap(testD['lWrist ori'][:, col], trainD['lWrist ori'][:, i:i + 1])))
                    Line = np.vstack(
                        (Line, logmap(testD['lShoulder ori'][:, col], trainD['lShoulder ori'][:, i:i + 1])))
                    Line = np.vstack((Line, logmap(testD['rElbow ori'][:, col], trainD['rElbow ori'][:, i:i + 1])))
                    Line = np.vstack((Line, logmap(testD['rWrist ori'][:, col], trainD['rWrist ori'][:, i:i + 1])))
                    Line = np.vstack(
                        (Line, logmap(testD['rShoulder ori'][:, col], trainD['rShoulder ori'][:, i:i + 1])))
                    dist = np.linalg.norm(Line, axis=0)
            prevDist = dist[n]
    # in Matlab, col==size(testD{1}.data,2 and poseTracking(keypose)==0 useless
    if keypose == m_cuts.size - 1 and prevDist < 1.4:
        poseTracking[keypose] = 1
        performedEntirely = 1

    # distance from init
    Line = logmap(testD['lElbow ori'][:, col], testD['lElbow ori'][:, 0:1])
    Line = np.vstack((Line, logmap(testD['lWrist ori'][:, col], testD['lWrist ori'][:, 0:1])))
    Line = np.vstack((Line, logmap(testD['lShoulder ori'][:, col], testD['lShoulder ori'][:, 0:1])))
    Line = np.vstack((Line, logmap(testD['rElbow ori'][:, col], testD['rElbow ori'][:, 0:1])))
    Line = np.vstack((Line, logmap(testD['rWrist ori'][:, col], testD['rWrist ori'][:, 0:1])))
    Line = np.vstack((Line, logmap(testD['rShoulder ori'][:, col], testD['rShoulder ori'][:, 0:1])))
    Line = np.linalg.norm(Line, axis=0)
    distFromInit = np.where(Line < 1.4, 0, 1)

    # poseTracking
    if np.sum(distFromInit)/distFromInit.size > 0.2:
        motionStarted = 1

    # dynamic programming
    p, q = dp(DM)  ## execution time : 1s
    # alignment
    r = np.zeros(DM.shape[1])
    for t in range(DM.shape[1]):
        ind = np.where(p == t)[0]
        if ind.size == 0:
            r[t] = p[t]
        else:
            r[t] = q[ind[0]]

    if performedEntirely == 1 and r[-1] < np.max(p):
        r[-2] = r[-3]
        r[-1] = r[-3]
    r = r.astype(int)
    r = row[r] #"row" equals "kf" in Matlab
    rfull = np.zeros(sizeData)
    rfull[row] = r

    if fast == 1:
        for i in range(r.size - 1):
            if r[i] == r[i+1]:
                temp = np.ones(resampleSize-1) * r[i]
            else:
                temp = round(np.linspace(r[i] + 1, r[i+1] - 1, resampleSize - 1))
            rfull[range(i*resampleSize+1, (i+1)*resampleSize)] = temp
        rfull = rfull.astype(int)
    r = rfull

    # no need to check r=0, this case is impossible

    out = {}
    for key in testD:
        out[key] = testD[key][:, r]
    return out, r, performedEntirely, poseTracking, motionStarted, distFromInit

