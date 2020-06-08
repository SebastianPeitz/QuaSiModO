import numpy as np_
import scipy as sp_
import d3s.observables as observables
import d3s.algorithms as algorithms


def timeTMap(z0, t0, iu, modelData):

    z0 = np_.array([z0]).transpose()
    z = modelData.K[:, :, iu] @ modelData.psi(z0)
    t = t0 + modelData.h
    return z[modelData.P, 0], t, modelData


def createSurrogateModel(modelData, data):

    if modelData.nMonomials == 0:
        psi = Identity
    else:
        psi = observables.monomials(modelData.nMonomials)

    X = data['X']
    Y = data['Y']
    for i in range(len(X)):
        # Ki, _, _ = algorithms.edmd(X[i].T, Y[i].T, modelData.psi, operator='K', evs=1)
        PsiX = psi(X[i].T)
        PsiY = psi(Y[i].T)
        # if PsiX.shape[0] > 1e4:
        #     if i == 0:
        #         chunkSize = 500
        #         nChunks = int(np_.ceil(PsiX.shape[0]/chunkSize))
        #     Gi = np_.zeros([PsiX.shape[0], PsiX.shape[0]], dtype=float)
        #     Ai = np_.zeros([PsiX.shape[0], PsiX.shape[0]], dtype=float)
        #     for i1 in range(nChunks):
        #         print(i1)
        #         in1, in2 = i1 * chunkSize, min((i1 + 1) * chunkSize, PsiX.shape[0])
        #         Gi[in1: in2, :] = PsiX[in1: in2, :] @ PsiX.T
        #         Ai[in1: in2, :] = PsiX[in1: in2, :] @ PsiY.T
        # else:
        Gi = PsiX @ PsiX.T
        Ai = PsiX @ PsiY.T

        Ki = sp_.linalg.pinv(Gi) @ Ai
        if i == 0:
            K = np_.zeros([Ki.shape[0], Ki.shape[1], len(X)], dtype=float)
            A = np_.zeros([Ki.shape[0], Ki.shape[1], len(X)], dtype=float)
            G = np_.zeros([Ki.shape[0], Ki.shape[1], len(X)], dtype=float)
            m = np_.zeros([len(X)], dtype=float)

        K[:, :, i] = Ki.transpose()
        A[:, :, i] = Ai
        G[:, :, i] = Gi
        m[i] = float(X[i].shape[0])

    if modelData.nMonomials == 0:
        P = np_.array(range(X[0].shape[1]))
    else:
        P = np_.array(range(1, X[0].shape[1] + 1))

    setattr(modelData, 'psi', psi)
    setattr(modelData, 'K', K)
    setattr(modelData, 'A', A)
    setattr(modelData, 'G', G)
    setattr(modelData, 'm', m)
    setattr(modelData, 'P', P)

    return modelData


def Identity(x):
    return x


def updateSurrogateModel(modelData, z, u, iu):
    dimZ = z.shape[1]
    updatePerformed = False
    for i in range(modelData.A.shape[2]):
        s = 0
        for j in range(iu.shape[0] - (modelData.nDelay + 1) * modelData.nLag):
            # if not any(iu[j: j + (modelData.nDelay + 1) * modelData.nLag] != i):
            if not any(iu[j + modelData.nDelay * modelData.nLag: j + (modelData.nDelay + 1) * modelData.nLag] != i):
                xi = np_.zeros([z.shape[0], (modelData.nDelay + 1) * dimZ], dtype=float)
                yi = np_.zeros(xi.shape, dtype=float)
                for k in range(modelData.nDelay + 1):
                    xi[s, k * dimZ: (k + 1) * dimZ] = z[j + (modelData.nDelay - k) * modelData.nLag, :]
                    yi[s, k * dimZ: (k + 1) * dimZ] = z[j + (modelData.nDelay - k + 1) * modelData.nLag, :]
                s += 1
        if s > 0:
            PsiX = modelData.psi(xi[:s, :].T)
            PsiY = modelData.psi(yi[:s, :].T)

            q = modelData.m[i] * modelData.epsUpdate / (1.0 - modelData.epsUpdate)
            modelData.A[:, :, i] = (modelData.m[i] * modelData.A[:, :, i] + q * (PsiX @ PsiY.T)) / (modelData.m[i] + q)
            modelData.G[:, :, i] = (modelData.m[i] * modelData.G[:, :, i] + q * (PsiX @ PsiY.T)) / (modelData.m[i] + q)
            modelData.K[:, :, i] = (sp_.linalg.pinv(modelData.G[:, :, i]) @ modelData.A[:, :, i]).transpose()
            modelData.m[i] += q

            updatePerformed = True

    return modelData, updatePerformed
