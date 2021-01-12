import numpy as np


def simulateModel(y0, t0, u, model):

    beta = 0.0084
    epsi = 0.0
    zeta = 0.0790
    lam = 0.0566
    theta = 0.1981
    kappa = 0.0563

    mu1, mu2 = 0.0080, 0.0050
    sigma1, sigma2 = 0.0370, 0.0552
    tau1, tau2 = 0.0159, 0.0242

    alphaMin, alphaMax = 0.0422, 0.3614
    gammaMin, gammaMax = 0.0422, 0.3614

    mu = mu1 + mu2
    sigma = mu1 / mu * sigma1 + mu2 / mu * sigma2
    tau = tau1 / mu * tau1 + mu2 / mu * tau2

    def rhs(y_, u_):
        alpha = alphaMax + u_[0] * (alphaMin - alphaMax)
        gamma = gammaMax + u_[0] * (gammaMin - gammaMax)

        dy = [-y_[0] * (alpha * y_[1] + beta * y_[2] + gamma * y_[3] + beta * y_[4]),
              y_[0] * (alpha * y_[1] + beta * y_[2] + gamma * y_[3] + beta * y_[4]) - (epsi + zeta + lam) * y_[1],
              epsi * y_[1] - (zeta + lam) * y_[2],
              zeta * y_[1] - (theta + mu + kappa) * y_[3],
              zeta * y_[2] + theta * y_[3] - (mu + kappa) * y_[4],
              mu * y_[3] + mu * y_[4] - (sigma + tau) * y_[5],
              lam * y_[1] + lam * y_[2] + kappa * y_[3] + kappa * y_[4] + sigma * y_[5],
              tau * y_[5]
              ]

        return np.array(dy)

    nt = u.shape[0]
    T = (nt - 1) * model.h
    t = np.linspace(0.0, T, nt) + t0
    y = np.empty([nt, len(y0)], dtype=float)
    y[0, :] = y0

    sigmaRand = np.zeros([1, 8], dtype=float)
    sigmaRand[0, 1] = 1.0 / 3.0

    for i in range(0, nt - 1):
        IDARTHE = 2.0
        while IDARTHE > 1.0:
            y[i + 1, :] = y[i, :] + model.h * rhs(y[i, :], u[i, :]) + np.sqrt(2.0 * model.h) * sigmaRand * y[i, :] * np.random.normal(0.0, 1.0, (1, 8))
            IDARTHE = sum(y[i + 1, 1:])
        y[i + 1, 0] = 1.0 - IDARTHE

    # Observation
    z = y[:, 1: -2]

    return y, z, t, model


def observable(y, model):
    return y[:, model.grid.iObs]


def calcJ(z, zRef, u, h, modelData):
    zeta = 0.0790
    lam = 0.0566
    kappa = 0.0563
    mu1, mu2 = 0.0080, 0.0050
    sigma1, sigma2 = 0.0370, 0.0552
    tau1, tau2 = 0.0159, 0.0242
    mu = mu1 + mu2
    sigma = mu1 / mu * sigma1 + mu2 / mu * sigma2
    tau = tau1 / mu * tau1 + mu2 / mu * tau2

    # Weights for the objective function
    c1 = zeta / (zeta + lam)
    c2 = mu / (mu + kappa)
    c3 = tau / (tau + sigma)

    ub_T = 40000.0 / 83000000.0

    # Q = np.array([c1 * c2 * c3, c1 * c2 * c3, c2 * c3, c2 * c3, c3])
    # J1 = 0.0
    # for i in range(z.shape[0]):
    #     J1 += np.sum(z[i, :] * Q * z[i, :])
    J1 = np.sum(np.sum(np.power(z, 2)))

    # # J2 = 1e-6 * np.maximum(1e-4, 1e-1 - np.mean(np.sum(z, axis=1))) * np.sum(u[:-1, :])/float(u.shape[0] - 1)
    J2 = 3e-2 * np.maximum(1e-4, 1e-1 - np.sqrt(np.mean(np.sum(z, axis=1)))) * np.sum(u[:-1, :])/float(u.shape[0] - 1)
    # if np.max(z[:, 0]) > 1e-3:
    #     J2 = 0.0
    # else:
    #     J2 = 1e-2 * np.sum(u[:-1, :]) / float(u.shape[0] - 1)

    penalty = 1e7*np.maximum(np.zeros([z.shape[0]], dtype=float), np.sign(z[:, -1] - ub_T) * np.power(z[:, -1] - ub_T, 2))
    J3 = np.sum(penalty)

    # print('J1 = ' + str(J1) + '; J2 = ' + str(J2) + '; J3 = ' + str(J3))

    return J1 + J2 + J3
