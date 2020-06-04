import numpy as np
import scipy as sp


def timeTMap(z0, t0, iu, modelData):
    alpha = modelData.Psi[iu].transpose() @ modelData.M2D @ (np.array([z0]).transpose() - modelData.yMean[iu])
    nt = 1

    aC = np.zeros(alpha.shape, dtype=float)
    for i in range(nt):

        a = alpha
        for j in range(len(aC)):
            aC[j, 0] = a.T @ modelData.C[iu][:, :, j] @ a
        k1 = modelData.A[iu] + modelData.B[iu] @ a + aC

        a = alpha + modelData.h * k1 / 2.0
        for j in range(len(aC)):
            aC[j, 0] = a.T @ modelData.C[iu][:, :, j] @ a
        k2 = modelData.A[iu] + modelData.B[iu] @ a + aC

        a = alpha + modelData.h * k2 / 2.0
        for j in range(len(aC)):
            aC[j, 0] = a.T @ modelData.C[iu][:, :, j] @ a
        k3 = modelData.A[iu] + modelData.B[iu] @ a + aC

        a = alpha + modelData.h * k3
        for j in range(len(aC)):
            aC[j, 0] = a.T @ modelData.C[iu][:, :, j] @ a
        k4 = modelData.A[iu] + modelData.B[iu] @ a + aC

        alpha += modelData.h / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
        # alpha += modelData.h * k1

    z = (modelData.Psi[iu] @ alpha) + modelData.yMean[iu]
    return z[:, 0], t0 + modelData.h, modelData


def calcJ(z, zRef, u, h, modelData):
    iu = mapUToIu(u, modelData.uGrid)
    alpha = np.zeros([iu.shape[0], modelData.A[0].shape[0]], dtype=float)
    for i in range(iu.shape[0]):
        alpha[i, :] = (z[i, :] - modelData.yMean[iu[i, 0]][:, 0]) @ modelData.M2D @ modelData.Psi[iu[i, 0]]
    deltaZ = np.array(alpha - zRef[: iu.shape[0], :modelData.A[0].shape[0]])
    deltaU = (u[1:, :] - u[:-1, :]) / h
    Q = np.identity(modelData.A[0].shape[0])
    R = np.array([[1e-4]])
    S = np.array([[0.0]])
    return np.trapz(np.einsum('ij,kj,ik->i', deltaZ, Q, deltaZ), dx=h) + \
           np.trapz(np.einsum('ij,kj,ik->i', u, R, u), dx=h) + \
           np.trapz(np.einsum('ij,kj,ik->i', deltaU, S, deltaU), dx=h)


def mapUToIu(u, uGrid):
    iu = -np.ones([u.shape[0], 1], dtype=int)
    for i in range(u.shape[0]):
        for j in range(uGrid.shape[0]):
            if all(u[i, :] == uGrid[j, :]):
                iu[i, 0] = j
                break

    return iu


def createSurrogateModel(modelData, data):
    V2D = np.zeros([2 * len(modelData.of.mesh.V)], dtype=float)
    V2D[::2] = modelData.of.mesh.V
    V2D[1::2] = modelData.of.mesh.V
    M2D = sp.sparse.spdiags(V2D, [0], len(V2D), len(V2D))
    Y = data['X']
    nSys = len(Y)
    dimZ = modelData.nModes + 1

    Psi = list()
    dPsi = list()
    dyMean = list()
    yMean = list()
    yShift = list()
    for i in range(nSys):
        yMean.append(np.zeros([2 * modelData.of.mesh.nc, 1], dtype=float))
        yMean[-1][:, 0] = np.mean(Y[i], axis=0)
        for j in range(Y[i].shape[0]):
            Y[i][j, :] -= yMean[i][:, 0]
        yShift.append(np.zeros([2 * modelData.of.mesh.nc, 1], dtype=float))
        yShift[-1][:, 0] = modelData.yBase[:, 0] - yMean[i][:, 0]

        D, V = sp.linalg.eig(Y[i] @ M2D @ Y[i].transpose())
        D = np.abs(D)
        I = np.argsort(-D)
        s = np.nan_to_num(np.sqrt(D[I]), nan=1e-12)
        U = Y[i].transpose() @ V[:, I[:modelData.nModes]]
        for j in range(U.shape[1]):
            U[:, j] /= s[j]

        print('POD model {} with {} modes contains {:.2f} % of information'.format(i, modelData.nModes, 100.0 * np.sum(
            s[:modelData.nModes]) / np.sum(s)))

        Psi.append(np.zeros([2 * modelData.of.mesh.nc, dimZ]))
        Psi[-1][:, 0] = yShift[i][:, 0]
        Psi[-1][:, 1:] = U[:, :modelData.nModes]

        for j in range(modelData.nModes):
            Psi[i][:, 0] -= (yShift[i].transpose() @ M2D @ Psi[i][:, j + 1]) * Psi[i][:, j + 1]
        Psi[i][:, 0] /= np.sqrt(Psi[i][:, 0].transpose() @ M2D @ Psi[i][:, 0])

        dyMean.append(modelData.of.calcGradient(yMean[i].T))
        dPsi.append(modelData.of.calcGradient(Psi[i].T))

        # modelData.of.writeField(Psi[i].T, 'U', iStart=200 + i * (modelData.nModes + 1))

    A, B, C = list(), list(), list()
    indices = np.asarray(range(0, 2 * modelData.of.mesh.nc, 2))
    for iSys in range(nSys):
        A.append(np.zeros([dimZ, 1], dtype=float))
        B.append(np.zeros([dimZ, dimZ], dtype=float))
        C.append(np.zeros([dimZ, dimZ, dimZ], dtype=float))
        for i in range(dimZ):
            A[-1][i, 0] = - (np.inner(modelData.of.mesh.V,
                                      (np.multiply(np.multiply(yMean[iSys][indices, 0], dyMean[iSys][0, :, 0, 0]) +
                                                   np.multiply(yMean[iSys][indices + 1, 0], dyMean[iSys][0, :, 0, 1]),
                                                   Psi[iSys][indices, i]) +
                                       np.multiply(np.multiply(yMean[iSys][indices, 0], dyMean[iSys][0, :, 1, 0]) +
                                                   np.multiply(yMean[iSys][indices + 1, 0], dyMean[iSys][0, :, 1, 1]),
                                                   Psi[iSys][indices + 1, i])
                                       ))) \
                          - (1.0 / modelData.Re) * (np.inner(modelData.of.mesh.V,
                                                             (np.multiply(dPsi[iSys][i, :, 0, 0],
                                                                          dyMean[iSys][0, :, 0, 0]) +
                                                              np.multiply(dPsi[iSys][i, :, 0, 1],
                                                                          dyMean[iSys][0, :, 0, 1]) +
                                                              np.multiply(dPsi[iSys][i, :, 1, 0],
                                                                          dyMean[iSys][0, :, 1, 0]) +
                                                              np.multiply(dPsi[iSys][i, :, 1, 1],
                                                                          dyMean[iSys][0, :, 1, 1])
                                                              )))
            for j in range(dimZ):
                B[-1][i, j] = - (np.inner(modelData.of.mesh.V,
                                          (np.multiply(np.multiply(yMean[iSys][indices, 0], dPsi[iSys][j, :, 0, 0]) +
                                                       np.multiply(yMean[iSys][indices + 1, 0], dPsi[iSys][j, :, 0, 1]),
                                                       Psi[iSys][indices, i]) +
                                           np.multiply(np.multiply(yMean[iSys][indices, 0], dPsi[iSys][j, :, 1, 0]) +
                                                       np.multiply(yMean[iSys][indices + 1, 0], dPsi[iSys][j, :, 1, 1]),
                                                       Psi[iSys][indices + 1, i])
                                           ))) \
                              - (np.inner(modelData.of.mesh.V,
                                          (np.multiply(np.multiply(Psi[iSys][indices, j], dyMean[iSys][0, :, 0, 0]) +
                                                       np.multiply(Psi[iSys][indices + 1, j], dyMean[iSys][0, :, 0, 1]),
                                                       Psi[iSys][indices, i]) +
                                           np.multiply(np.multiply(Psi[iSys][indices, j], dyMean[iSys][0, :, 1, 0]) +
                                                       np.multiply(Psi[iSys][indices + 1, j], dyMean[iSys][0, :, 1, 1]),
                                                       Psi[iSys][indices + 1, i])
                                           ))) \
                              - (1.0 / modelData.Re) * (np.inner(modelData.of.mesh.V,
                                                                 (np.multiply(dPsi[iSys][i, :, 0, 0],
                                                                              dPsi[iSys][j, :, 0, 0]) +
                                                                  np.multiply(dPsi[iSys][i, :, 0, 1],
                                                                              dPsi[iSys][j, :, 0, 1]) +
                                                                  np.multiply(dPsi[iSys][i, :, 1, 0],
                                                                              dPsi[iSys][j, :, 1, 0]) +
                                                                  np.multiply(dPsi[iSys][i, :, 1, 1],
                                                                              dPsi[iSys][j, :, 1, 1])
                                                                  )))
                for k in range(dimZ):
                    C[-1][j, k, i] = - (np.inner(modelData.of.mesh.V,
                                                 (np.multiply(
                                                     np.multiply(Psi[iSys][indices, j], dPsi[iSys][k, :, 0, 0]) +
                                                     np.multiply(Psi[iSys][indices + 1, j], dPsi[iSys][k, :, 0, 1]),
                                                     Psi[iSys][indices, i]) +
                                                  np.multiply(
                                                      np.multiply(Psi[iSys][indices, j], dPsi[iSys][k, :, 1, 0]) +
                                                      np.multiply(Psi[iSys][indices + 1, j], dPsi[iSys][k, :, 1, 1]),
                                                      Psi[iSys][indices + 1, i])
                                                  )))

    setattr(modelData, 'A', A)
    setattr(modelData, 'B', B)
    setattr(modelData, 'C', C)
    setattr(modelData, 'M2D', M2D)
    setattr(modelData, 'Psi', Psi)
    setattr(modelData, 'yMean', yMean)

    return modelData


def updateSurrogateModel(modelData, z, u, iu):
    updatePerformed = False
    return modelData, updatePerformed
