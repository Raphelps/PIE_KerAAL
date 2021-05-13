import numpy as np
from functions.mapping import logmap
from functions.Nodes import Lnode
from m_fcts.gaussPDF import gaussPDF


def computeLikelihoods(m_dt, m_nbStates, m_nbVar, m_Priors, m_Mu, m_Sigma, m_MuMan, dataTest, sizeData):
    xIn = np.arange(1, sizeData + 1) * m_dt
    L = np.zeros((m_nbStates, sizeData))
    Data_gaussPDF = np.zeros((m_nbVar, sizeData, m_nbStates))
    Lglobal = {}
    Lbodypart = {}
    Ljoints = {}

    # Global(orientation + position)
    for i in range(m_nbStates):
        Data_gaussPDF[0, :, i] = xIn - m_MuMan[0, i]
        Data_gaussPDF[1:4, :, i] = logmap(dataTest['lElbow ori'], m_MuMan[1:5, i])
        Data_gaussPDF[4:7, :, i] = logmap(dataTest['lWrist ori'], m_MuMan[5:9, i])
        Data_gaussPDF[7:10, :, i] = logmap(dataTest['lShoulder ori'], m_MuMan[9:13, i])
        Data_gaussPDF[10:13, :, i] = logmap(dataTest['rElbow ori'], m_MuMan[13:17, i])
        Data_gaussPDF[13:16, :, i] = logmap(dataTest['rWrist ori'], m_MuMan[17:21, i])
        Data_gaussPDF[16:19, :, i] = logmap(dataTest['rShoulder ori'], m_MuMan[21:25, i])
        Data_gaussPDF[19:22, :, i] = logmap(dataTest['mSpine ori'], m_MuMan[25:29, i])
        Data_gaussPDF[22:25, :, i] = logmap(dataTest['mShoulder ori'], m_MuMan[29:33, i])
        Data_gaussPDF[25:28, :, i] = logmap(dataTest['Neck ori'], m_MuMan[33:37, i])
        Data_gaussPDF[28:31, :, i] = dataTest['lElbow rel_pos'] - m_MuMan[37:40, i:i + 1]
        Data_gaussPDF[31:34, :, i] = dataTest['lWrist rel_pos'] - m_MuMan[40:43, i:i + 1]
        Data_gaussPDF[34:37, :, i] = dataTest['lShoulder rel_pos'] - m_MuMan[43:46, i:i + 1]
        Data_gaussPDF[37:40, :, i] = dataTest['rElbow rel_pos'] - m_MuMan[46:49, i:i + 1]
        Data_gaussPDF[40:43, :, i] = dataTest['rWrist rel_pos'] - m_MuMan[49:52, i:i + 1]
        Data_gaussPDF[43:46, :, i] = dataTest['rShoulder rel_pos'] - m_MuMan[52:55, i:i + 1]

        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[:, :, i], m_Mu[:, i], m_Sigma[:, :, i])
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    Lglobal['Global'] = Lnode(LL, score)

    # Orientations
    # out = 2:28 omitted to have a faster calulation
    sigma = np.zeros((28, 28))
    mu = np.zeros(28)
    for i in range(m_nbStates):
        mu[:] = m_Mu[:28, i]
        sigma[:, :] = m_Sigma[:28, :28, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[:28, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    Lglobal['Orientations'] = Lnode(LL, score)

    # Positions
    # out = 29:46 omitted to have a faster calulation
    sigma = np.zeros((19, 19))
    mu = np.zeros(19)
    index = [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    for i in range(m_nbStates):
        mu[:] = m_Mu[index, i]
        sigma[0, :] = m_Sigma[0, index, i]
        sigma[1:, :] = m_Sigma[28:46, index, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    Lglobal['Positions'] = Lnode(LL, score)

    # Body Part
    left_arm = {}
    right_arm = {}
    # Left Arm Global
    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    for i in range(m_nbStates):
        mu[:] = m_Mu[index, i]
        sigma[:10, :] = m_Sigma[:10, index, i]
        sigma[10:, :] = m_Sigma[28:37, index, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    left_arm['Left Arm Global'] = Lnode(LL, score)

    # Right Arm Global
    index = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    for i in range(m_nbStates):
        mu[:] = m_Mu[index, i]
        sigma[0, :] = m_Sigma[0, index, i]
        sigma[1:10, :] = m_Sigma[10:19, index, i]
        sigma[10:, :] = m_Sigma[37:46, index, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    right_arm['Right Arm Global'] = Lnode(LL, score)

    # Left Arm Orientation
    sigma = np.zeros((10, 10))
    mu = np.zeros(10)
    for i in range(m_nbStates):
        mu[:] = m_Mu[:10, i]
        sigma[:, :] = m_Sigma[:10, :10, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[:10, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    left_arm['Left Arm Orientation'] = Lnode(LL, score)

    # Left Arm Postion
    index = [0, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    for i in range(m_nbStates):
        mu[:] = m_Mu[index, i]
        sigma[0, :] = m_Sigma[0, index, i]
        sigma[1:, :] = m_Sigma[28:37, index, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    left_arm['Left Arm Position'] = Lnode(LL, score)

    # Right Arm Orientation
    index = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for i in range(m_nbStates):
        mu[:] = m_Mu[index, i]
        sigma[0, :] = m_Sigma[0, index, i]
        sigma[1:, :] = m_Sigma[10:19, index, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    right_arm['Right Arm Orientation'] = Lnode(LL, score)

    # Right Arm Position
    index = [0, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    for i in range(m_nbStates):
        mu[:] = m_Mu[index, i]
        sigma[0, :] = m_Sigma[0, index, i]
        sigma[1:, :] = m_Sigma[37:46, index, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    right_arm['Right Arm Position'] = Lnode(LL, score)

    Lbodypart['Left Arm'] = left_arm
    Lbodypart['Right Arm'] = right_arm

    # Colonne Global (only position)
    Colonne = {}
    index = [0, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    for i in range(m_nbStates):
        mu[:] = m_Mu[index, i]
        sigma[0, :] = m_Sigma[0, index, i]
        sigma[1:, :] = m_Sigma[19:28, index, i]
        L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
    LL = np.log(np.sum(L, axis=0))
    LL = np.where(LL < -2000, -2000, LL)
    score = np.mean(LL)
    Colonne['Colonne Global'] = Lnode(LL, score)
    Lbodypart['Colonne'] = Colonne

    # Per Joints
    oriL = np.zeros((m_nbStates, sizeData))
    posL = np.zeros((m_nbStates, sizeData))
    jointNames1 = ['lElbow', 'lWrist', 'lShoulder', 'rElbow', 'rWrist', 'rShoulder']
    jointNames2 = ['mSpine', 'mShoulder', 'Neck']
    index = np.array([0, 1, 2, 3, 28, 29, 30])
    a = [0, 4, 5, 6]
    sigma = np.zeros((7, 7))
    mu = np.zeros(7)
    sigma_ = np.zeros((4,4))
    mu_ = np.zeros(4)

    # first six
    for name in jointNames1:
        tmp = {}
        for i in range(m_nbStates):
            mu[:] = m_Mu[index, i]
            sigma[:, :] = m_Sigma[np.ix_(index, index, [i])].reshape((7,7))
            # Global
            L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu, sigma)
            # Orientations
            mu_[:] = mu[:4]
            sigma_[:] = sigma[:4, :4]
            oriL[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index[:4], :, i], mu_, sigma_)
            # Positions
            index_ = index[a]
            mu_[:] = mu[a]
            sigma_[:] = sigma[np.ix_(a, a)]
            posL[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index_, :, i], mu_, sigma_)
        # Global
        LL = np.log(np.sum(L, axis=0))
        LL = np.where(LL < -2000, -2000, LL)
        score = np.mean(LL)
        tmp[name+' Global'] = Lnode(LL, score)
        # Orientations
        LL = np.log(np.sum(oriL, axis=0))
        LL = np.where(LL < -2000, -2000, LL)
        score = np.mean(LL)
        tmp[name+' Orientations'] = Lnode(LL, score)
        # Positions
        LL = np.log(np.sum(posL, axis=0))
        LL = np.where(LL < -2000, -2000, LL)
        score = np.mean(LL)
        tmp[name+' Positions'] = Lnode(LL, score)
        Ljoints[name] = tmp.copy()

        index += 3
        index[0] -= 3

    # last three
    index = index[:4]
    for name in jointNames2:
        tmp = {}
        for i in range(m_nbStates):
            mu_[:] = m_Mu[index, i]
            sigma_[:, :] = m_Sigma[np.ix_(index, index, [i])].reshape((4,4))
            L[i, :] = m_Priors[i] * gaussPDF(Data_gaussPDF[index, :, i], mu_, sigma_)
        # Global
        LL = np.log(np.sum(L, axis=0))
        LL = np.where(LL < -2000, -2000, LL)
        score = np.mean(LL)
        tmp[name+' Global'] = Lnode(LL, score)
        Ljoints[name] = tmp.copy()

        index += 3
        index[0] -= 3

    return Lglobal, Lbodypart, Ljoints
