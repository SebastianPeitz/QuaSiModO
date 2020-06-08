import numpy as np_
import scipy as sp_
import d3s.observables as observables


def timeTMap(z0, t0, iu, modelData):

    z = modelData.psi(np_.array([z0]).transpose())
    k1 = modelData.K[:, :, iu] @ z
    k2 = modelData.K[:, :, iu] @ (z + 0.5 * modelData.h * k1)
    k3 = modelData.K[:, :, iu] @ (z + 0.5 * modelData.h * k2)
    k4 = modelData.K[:, :, iu] @ (z + modelData.h * k3)
    z += modelData.h / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    t = t0 + modelData.h
    return z[1: 1 + z0.shape[0], 0], t, modelData


def createSurrogateModel(modelData, data):
    psi = observables.monomials(modelData.nMonomials)
    X = data['X']
    Y = data['dX']

    for i in range(len(X)):
        PsiX = psi(X[i].T)
        dPsiY = np_.einsum('ijk,jk->ik', psi.diff(X[i].T), Y[i].T)
        Gi = PsiX @ PsiX.T
        Ai = PsiX @ dPsiY.T

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

    setattr(modelData, 'psi', psi)
    setattr(modelData, 'K', K)
    setattr(modelData, 'A', A)
    setattr(modelData, 'G', G)
    setattr(modelData, 'm', m)

    return modelData


# def updateSurrogateModel(modelData, z, u, iu):
#     dimZ = z.shape[1]
#     updatePerformed = False
#     for i in range(modelData.A.shape[2]):
#         s = 0
#         for j in range(iu.shape[0] - (modelData.nDelay + 1) * modelData.nLag):
#             xi = np_.zeros([z.shape[0], (modelData.nDelay + 1) * dimZ], dtype=float)
#             yi = np_.zeros(xi.shape, dtype=float)
#             if not any(iu[j: j + (modelData.nDelay + 1) * modelData.nLag - 1] != i):
#                 for k in range(modelData.nDelay + 1):
#                     xi[s, k * dimZ: (k + 1) * dimZ] = z[j + (modelData.nDelay - k) * modelData.nLag, :]
#                     yi[s, k * dimZ: (k + 1) * dimZ] = z[j + (modelData.nDelay - k + 1) * modelData.nLag, :]
#                 s += 1
#         if s > 0:
#             PsiX = modelData.psi(xi[:s, :].T)
#             PsiY = modelData.psi(yi[:s, :].T)
#
#             q = modelData.m[i] * modelData.epsUpdate / (1.0 - modelData.epsUpdate)
#             modelData.A[:, :, i] = (modelData.m[i] * modelData.A[:, :, i] + q * (PsiX @ PsiY.T)) / (modelData.m[i] + q)
#             modelData.G[:, :, i] = (modelData.m[i] * modelData.G[:, :, i] + q * (PsiX @ PsiY.T)) / (modelData.m[i] + q)
#             modelData.K[:, :, i] = (sp_.linalg.pinv(modelData.G[:, :, i]) @ modelData.A[:, :, i]).transpose()
#             modelData.m[i] += q
#
#             updatePerformed = True
#
#     return modelData, updatePerformed
