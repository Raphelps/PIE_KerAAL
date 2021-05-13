import numpy as np
from m_fcts.init_GMM_kbins import init_GMM_kbins
from m_fcts.gaussPDF import gaussPDF
from functions.mapping import logmap, expmap


def learnGMMmodel(model, u, xIn, xOut, nbSamples, nbIterEM, nbIter, nbData):
    init_GMM_kbins(u, model, nbSamples)
    # MuMan
    mu_exp = np.array([[0], [1], [0], [0]])
    model.MuMan = np.zeros((model.nbVarMan, model.nbStates))
    model.MuMan[0, :] = model.Mu[0, :]
    model.MuMan[1:5, :] = expmap(model.Mu[1:4], mu_exp)
    model.MuMan[5:9, :] = expmap(model.Mu[4:7], mu_exp)
    model.MuMan[9:13, :] = expmap(model.Mu[7:10], mu_exp)
    model.MuMan[13:17, :] = expmap(model.Mu[10:13], mu_exp)
    model.MuMan[17:21, :] = expmap(model.Mu[13:16], mu_exp)
    model.MuMan[21:25, :] = expmap(model.Mu[16:19], mu_exp)
    model.MuMan[25:29, :] = expmap(model.Mu[19:22], mu_exp)
    model.MuMan[29:33, :] = expmap(model.Mu[22:25], mu_exp)
    model.MuMan[33:37, :] = expmap(model.Mu[25:28], mu_exp)
    model.MuMan[37:55, :] = model.Mu[28:46, :]
    # Mu
    model.Mu = np.zeros((model.nbVar, model.nbStates)) ## ???

    uTmp = np.zeros((model.nbVar, nbData * nbSamples, model.nbStates))
    avg_loglik = np.zeros(nbIterEM)
    Data_gaussPDF = np.zeros((model.nbVar, xIn.size))  ## or xIn.shape[1]
    U0 = np.zeros((model.nbVar, model.nbVar, model.nbStates),dtype=complex)  ## U0 is complex
    for nb in range(nbIterEM):
        L = np.zeros((model.nbStates, model.x.shape[1]))
        for i in range(model.nbStates):
            Data_gaussPDF[0, :] = xIn - model.MuMan[0, i]
            Data_gaussPDF[1:4, :] = logmap(xOut[1], model.MuMan[1:5, i])
            Data_gaussPDF[4:7, :] = logmap(xOut[2], model.MuMan[5:9, i])
            Data_gaussPDF[7:10, :] = logmap(xOut[3], model.MuMan[9:13, i])
            Data_gaussPDF[10:13, :] = logmap(xOut[4], model.MuMan[13:17, i])
            Data_gaussPDF[13:16, :] = logmap(xOut[5], model.MuMan[17:21, i])
            Data_gaussPDF[16:19, :] = logmap(xOut[6], model.MuMan[21:25, i])
            Data_gaussPDF[19:22, :] = logmap(xOut[7], model.MuMan[25:29, i])
            Data_gaussPDF[22:25, :] = logmap(xOut[8], model.MuMan[29:33, i])
            Data_gaussPDF[25:28, :] = logmap(xOut[9], model.MuMan[33:37, i])
            Data_gaussPDF[28:31, :] = xOut[10] - model.MuMan[37:40, i:i + 1]
            Data_gaussPDF[31:34, :] = xOut[11] - model.MuMan[40:43, i:i + 1]
            Data_gaussPDF[34:37, :] = xOut[12] - model.MuMan[43:46, i:i + 1]
            Data_gaussPDF[37:40, :] = xOut[13] - model.MuMan[46:49, i:i + 1]
            Data_gaussPDF[40:43, :] = xOut[14] - model.MuMan[49:52, i:i + 1]
            Data_gaussPDF[43:46, :] = xOut[15] - model.MuMan[52:55, i:i + 1]

            L[i, :] = model.Priors[i] * gaussPDF(Data_gaussPDF, model.Mu[:, i], model.Sigma[:, :, i])

        GAMMA = L / (np.sum(L, axis=0, keepdims=True) + np.finfo(float).tiny)
        GAMMA2 = GAMMA / np.sum(GAMMA, axis=1, keepdims=True)
        GAMMA2 = np.where(np.isnan(GAMMA2), 0, GAMMA2)
        avg_loglik[nb] = -np.log(np.mean(np.sum(L, axis=0)))
        # M-step
        for i in range(model.nbStates):
            # Update Priors
            model.Priors[i] = np.sum(GAMMA[i, :]) / (nbData * nbSamples)
            # Update MuMan
            for n in range(nbIter):
                uTmp[0, :, i] = xIn - model.MuMan[0, i]
                uTmp[1:4, :, i] = logmap(xOut[1], model.MuMan[1:5, i])
                uTmp[4:7, :, i] = logmap(xOut[2], model.MuMan[5:9, i])
                uTmp[7:10, :, i] = logmap(xOut[3], model.MuMan[9:13, i])
                uTmp[10:13, :, i] = logmap(xOut[4], model.MuMan[13:17, i])
                uTmp[13:16, :, i] = logmap(xOut[5], model.MuMan[17:21, i])
                uTmp[16:19, :, i] = logmap(xOut[6], model.MuMan[21:25, i])
                uTmp[19:22, :, i] = logmap(xOut[7], model.MuMan[25:29, i])
                uTmp[22:25, :, i] = logmap(xOut[8], model.MuMan[29:33, i])
                uTmp[25:28, :, i] = logmap(xOut[9], model.MuMan[33:37, i])
                uTmp[28:31, :, i] = xOut[10] - model.MuMan[37:40, i:i + 1]
                uTmp[31:34, :, i] = xOut[11] - model.MuMan[40:43, i:i + 1]
                uTmp[34:37, :, i] = xOut[12] - model.MuMan[43:46, i:i + 1]
                uTmp[37:40, :, i] = xOut[13] - model.MuMan[46:49, i:i + 1]
                uTmp[40:43, :, i] = xOut[14] - model.MuMan[49:52, i:i + 1]
                uTmp[43:46, :, i] = xOut[15] - model.MuMan[52:55, i:i + 1]

                GAMMA2_col = GAMMA2[i:i + 1, :].T
                model.MuMan[0, i] = np.dot(model.MuMan[0, i] + uTmp[0, :, i], GAMMA2_col)
                model.MuMan[1:5, i:i + 1] = expmap(np.dot(uTmp[1:4, :, i], GAMMA2_col), model.MuMan[1:5, i])
                model.MuMan[5:9, i:i + 1] = expmap(np.dot(uTmp[4:7, :, i], GAMMA2_col), model.MuMan[5:9, i])
                model.MuMan[9:13, i:i + 1] = expmap(np.dot(uTmp[7:10, :, i], GAMMA2_col), model.MuMan[9:13, i])
                model.MuMan[13:17, i:i + 1] = expmap(np.dot(uTmp[10:13, :, i], GAMMA2_col), model.MuMan[13:17, i])
                model.MuMan[17:21, i:i + 1] = expmap(np.dot(uTmp[13:16, :, i], GAMMA2_col), model.MuMan[17:21, i])
                model.MuMan[21:25, i:i + 1] = expmap(np.dot(uTmp[16:19, :, i], GAMMA2_col), model.MuMan[21:25, i])
                model.MuMan[25:29, i:i + 1] = expmap(np.dot(uTmp[19:22, :, i], GAMMA2_col), model.MuMan[25:29, i])
                model.MuMan[29:33, i:i + 1] = expmap(np.dot(uTmp[22:25, :, i], GAMMA2_col), model.MuMan[29:33, i])
                model.MuMan[33:37, i:i + 1] = expmap(np.dot(uTmp[25:28, :, i], GAMMA2_col), model.MuMan[33:37, i])
                model.MuMan[37:40, i:i + 1] = np.dot(model.MuMan[37:40, i:i + 1] + uTmp[28:31, :, i], GAMMA2_col)
                model.MuMan[40:43, i:i + 1] = np.dot(model.MuMan[40:43, i:i + 1] + uTmp[31:34, :, i], GAMMA2_col)
                model.MuMan[43:46, i:i + 1] = np.dot(model.MuMan[43:46, i:i + 1] + uTmp[34:37, :, i], GAMMA2_col)
                model.MuMan[46:49, i:i + 1] = np.dot(model.MuMan[46:49, i:i + 1] + uTmp[37:40, :, i], GAMMA2_col)
                model.MuMan[49:52, i:i + 1] = np.dot(model.MuMan[49:52, i:i + 1] + uTmp[40:43, :, i], GAMMA2_col)
                model.MuMan[52:55, i:i + 1] = np.dot(model.MuMan[52:55, i:i + 1] + uTmp[43:46, :, i], GAMMA2_col)
            # Update Sigma
            model.Sigma[:, :, i] = uTmp[:, :, i].dot(np.diag(GAMMA2[i, :])).dot(uTmp[:, :, i].T) + np.identity(
                u.shape[0]) * model.params_diagRegFact

    # Eigendecomposition of Sigma
    for i in range(model.nbStates):
        W, V = np.linalg.eig(model.Sigma[:, :, i]) ##np.linalg.eig doesn't return exactly the same result as eig() in matlab
        D_sqrt = np.diag(np.sqrt(W))
        U0[:, :, i] = np.dot(V, D_sqrt)
    model.U0 = U0
    pass
