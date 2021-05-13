import numpy as np


def estimateOrientationFromPosition(posMat):
    # defined bases for each joint
    bases = np.array([[0, 0, 0],  # bSpine
                      [0, 1, 0],  # mSpine
                      [0, 1, 0],  # Neck
                      [0, 1, 0],  # Head
                      [0, 1, 0],  # lShoulder
                      [0, 1, 0],  # lElbow
                      [0, 1, 0],  # lWrist
                      [0, 1, 0],  # lHand
                      [0, 1, 0],  # rShoulder
                      [0, 1, 0],  # rElbow
                      [0, 1, 0],  # rWrist
                      [0, 1, 0],  # rHand
                      [0, 1, 0],  # lHip
                      [0, 1, 0],  # lKnee
                      [0, 1, 0],  # lAnkle
                      [0, 1, 0],  # lFoot
                      [0, 1, 0],  # rHip
                      [0, 1, 0],  # rKnee
                      [0, 1, 0],  # rAnkle
                      [0, 1, 0],  # rFoot
                      [0, 1, 0],  # mShoulder
                      [0, 1, 0],  # lTip
                      [0, 1, 0],  # lThumb
                      [0, 1, 0],  # rTip
                      [0, 1, 0]  # rThumb
                      ])
    dirs = np.zeros(posMat.shape)
    dirs[:, 0:3] = np.zeros((posMat.shape[0], 3))  # bSpine
    dirs[:, 3:6] = posMat[:, 3:6] - posMat[:, 0:3]  # bSpine
    dirs[:, 6:9] = posMat[:, 6:9] - posMat[:, 60:63]  # bSpine
    dirs[:, 9:12] = posMat[:, 9:12] - posMat[:, 6:9]  # Head
    dirs[:, 12:15] = posMat[:, 12:15] - posMat[:, 60:63]  # lShoulder
    dirs[:, 15:18] = posMat[:, 15:18] - posMat[:, 12:15]  # lElbow
    dirs[:, 18:21] = posMat[:, 18:21] - posMat[:, 15:18]  # lWrist
    dirs[:, 21:24] = posMat[:, 21:24] - posMat[:, 18:21]  # lHand
    dirs[:, 24:27] = posMat[:, 24:27] - posMat[:, 60:63]  # rShoulder
    dirs[:, 27:30] = posMat[:, 27:30] - posMat[:, 24:27]  # rElbow
    dirs[:, 30:33] = posMat[:, 30:33] - posMat[:, 27:30]  # rWrist
    dirs[:, 33:36] = posMat[:, 33:36] - posMat[:, 30:33]  # rHand
    dirs[:, 36:39] = posMat[:, 36:39] - posMat[:, 0:3]  # lHip
    dirs[:, 39:42] = posMat[:, 39:42] - posMat[:, 36:39]  # lKnee
    dirs[:, 42:45] = posMat[:, 42:45] - posMat[:, 39:42]  # lAnkle
    dirs[:, 45:48] = posMat[:, 45:48] - posMat[:, 42:45]  # lFoot
    dirs[:, 48:51] = posMat[:, 48:51] - posMat[:, 0:3]  # rHip
    dirs[:, 51:54] = posMat[:, 51:54] - posMat[:, 48:51]  # rKnee
    dirs[:, 54:57] = posMat[:, 54:57] - posMat[:, 51:54]  # rAnkle
    dirs[:, 57:60] = posMat[:, 57:60] - posMat[:, 54:57]  # rFoot
    dirs[:, 60:63] = posMat[:, 60:63] - posMat[:, 3:6]  # mShoulder
    dirs[:, 63:66] = posMat[:, 63:66] - posMat[:, 18:21]  # lTip
    dirs[:, 66:69] = posMat[:, 66:69] - posMat[:, 21:24]  # lThumb
    dirs[:, 69:72] = posMat[:, 69:72] - posMat[:, 30:33]  # rTip
    dirs[:, 72:75] = posMat[:, 72:75] - posMat[:, 33:36]  # rThumb
    oriMat = np.array([])
    for t in range(posMat.shape[0]):
        oriVec = np.zeros(100)  ##1 -dimension vector
        for b in range(25):
            fO = b * 4
            fP = b * 3
            oriVec[fO:fO + 4] = compute_q_from_dirbase(dirs[t, fP:fP + 3], bases[b, :])
        if t == 0:
            oriVec = np.array([oriVec])
            oriMat = oriVec.T
        else:
            oriVec = np.array([oriVec])
            oriMat = np.hstack((oriMat, oriVec.T))
    oriMat = np.where(np.isnan(oriMat), 0, oriMat)
    return oriMat.T


def compute_q_from_dirbase(dir, base):  ##return 1-dimension vector
    normD = np.linalg.norm(dir)
    dir = dir / normD
    a = np.cross(dir, base)
    w = np.dot(dir, base)
    q = np.array([w, a[0], a[1], a[2]])
    normq = np.linalg.norm(q)
    q[0] = q[0] + normq
    normq = np.linalg.norm(q)
    q = q / normq
    return q
