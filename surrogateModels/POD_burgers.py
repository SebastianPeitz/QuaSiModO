import numpy as np
import scipy as sp
from visualization import *


def timeTMap(z0, t0, iu, modelData):

    alpha = modelData.Psi[iu].transpose() @ np.array([z0]).transpose()

    nt = 1

    # aB = np.zeros(alpha.shape, dtype=float)
    for i in range(nt):

        a = alpha
        k1 = -(modelData.A[iu] / modelData.Re + evalT(modelData.B[iu], a)) @ a + modelData.C[iu]

        a = alpha + modelData.h * k1 / 2.0
        k2 = -(modelData.A[iu] / modelData.Re + evalT(modelData.B[iu], a)) @ a + modelData.C[iu]

        a = alpha + modelData.h * k2 / 2.0
        k3 = -(modelData.A[iu] / modelData.Re + evalT(modelData.B[iu], a)) @ a + modelData.C[iu]

        a = alpha + modelData.h * k3
        k4 = -(modelData.A[iu] / modelData.Re + evalT(modelData.B[iu], a)) @ a + modelData.C[iu]

        alpha += modelData.h / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
        # alpha += modelData.h * k1

    z = modelData.Psi[iu] @ alpha
    return z[:, 0], t0 + modelData.h, modelData


def evalT(B, a):
    aB = np.zeros([a.shape[0], a.shape[0]], dtype=float)
    for j in range(a.shape[0]):
        aB += a[j, 0] * B[:, :, j]
    return aB


def createSurrogateModel(modelData, data):

    Y = data['X']
    # for i in range(len(Y)):
    #     Y[i] = Y[i][::10, :]

    nSys = len(Y)
    nx = Y[0].shape[1]
    dimZ = modelData.nModes

    Psi = list()
    dPsi = list()

    for i in range(nSys):

        U, s, Vh = sp.linalg.svd(Y[i].transpose())

        print('POD model {} with {} modes contains {:.2f} % of information'.format(i, modelData.nModes, 100.0 * np.sum(
            s[:modelData.nModes]) / np.sum(s)))

        Psi.append(U[:, :modelData.nModes])

        dPsi.append(np.zeros(Psi[-1].shape))
        dPsi[-1][0, :] = (Psi[-1][1, :] - Psi[-1][0, :]) / modelData.dx
        dPsi[-1][-1, :] = (Psi[-1][-1, :] - Psi[-1][-2, :]) / modelData.dx
        dPsi[-1][1:-1, :] = (Psi[-1][2:, :] - Psi[-1][: -2, :]) / (2.0 * modelData.dx)

        # plot(Psi={'t': np.arange(0, 1.0 + modelData.dx, modelData.dx), 'Psi': Psi[-1], 'iplot': 0},
        #      dPsi={'t': np.arange(0, 1.0 + modelData.dx, modelData.dx), 'dPsi': dPsi[-1], 'iplot': 1})

    A, B, C = list(), list(), list()
    for iSys in range(nSys):
        A.append(np.zeros([dimZ, dimZ], dtype=float))
        B.append(np.zeros([dimZ, dimZ, dimZ], dtype=float))
        C.append(np.zeros([dimZ, 1], dtype=float))
        for i in range(dimZ):
            for j in range(len(modelData.Chi_u)):
                C[-1][i, 0] += (modelData.uGrid[iSys][j] * modelData.Chi_u[j]).transpose() @ Psi[iSys][:, i]
            # C[-1][i, 0] = (modelData.uGrid[iSys][0] * Chi_u1 + modelData.uGrid[iSys][1] * Chi_u2 + modelData.uGrid[iSys][2] * Chi_u3).transpose() @ Psi[iSys][:, i]
            for j in range(dimZ):
                A[-1][i, j] = dPsi[iSys][:, i].transpose() @ dPsi[iSys][:, j]

                for k in range(dimZ):
                    B[-1][i, j, k] = np.sum(Psi[iSys][:, j] * dPsi[iSys][:, k] * Psi[iSys][:, i])

    setattr(modelData, 'A', A)
    setattr(modelData, 'B', B)
    setattr(modelData, 'C', C)
    setattr(modelData, 'Psi', Psi)

    return modelData


def updateSurrogateModel(modelData, z, u, iu):
    updatePerformed = False
    return modelData, updatePerformed
