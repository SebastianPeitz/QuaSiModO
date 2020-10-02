import numpy as np
# import matplotlib.pyplot as plt


def createChi(model):
    nx = model.grid.x.shape[0]
    # nChi = int(np.floor(nx/2.0))
    nChi = int(np.floor(float(nx)/float(np.maximum(model.dimU, 2))))
    iu = np.zeros([model.dimU, 2], dtype=int)
    for i in range(model.dimU):
        iu[i, 0] = i * int(np.floor(float(nx - nChi) / float(np.maximum(model.dimU, 2) - 1)))
        iu[i, 1] = iu[i, 0] + nChi
    dL = nx - (iu[-1, 1] + 1)
    if dL < 0:
        iu[i, 0] += dL
        iu[i, 1] += dL
    xSin = np.linspace(-0.5 * np.pi, 1.5 * np.pi, iu[0, 1] - iu[0, 0] + 1)
    Chi_u = list()
    for i in range(model.dimU):
        Chi_u.append(np.zeros(nx, dtype=float))
        # Chi_u[i][iu[i, 0]: iu[i, 1] + 1] = 0.5 * (np.sin(xSin) + 1.0)
        Chi_u[i][iu[i, 0]: iu[i, 1] + 1] = 1.0

    return Chi_u


def simulateModel(y0, t0, u, model):

    flagDirichlet0 = model.params['flagDirichlet0']

    viscosity = 1.0 / model.params['Re']

    # Problem dimensions
    nt = u.shape[0]
    nx = model.grid.x.shape[0]
    t = np.linspace(t0, t0 + nt * model.h, nt)

    # Setting up indices of neighbors for finite differencing
    ip = np.append(np.arange(1, nx), [0])
    im = np.arange(-1, nx - 1)

    # Initial condition
    y = np.zeros([nt, nx], dtype=float)
    y[0, :] = y0

    if flagDirichlet0:
        y[0, 0] = 0.0
        y[0, -1] = 0.0

    # Distributed control input via characteristic function
    Chi_u = createChi(model)
    # fig = plt.figure()
    # for i in range(len(Chi_u)):
    #     plt.plot(model.grid.x, Chi_u[i])
    # plt.grid(True)
    # plt.show()

    # Time integration
    def rhs(y_, u_):
        ydot = -(y_ * (y_[ip] - y_[im]) / (2.0 * model.grid.dx)) + \
               (viscosity * (y_[ip] - 2.0 * y_ + y_[im]) / (model.grid.dx * model.grid.dx))
        for i_ in range(model.dimU):
            ydot += (Chi_u[i_] * u_[i_])
        if flagDirichlet0:
            ydot[0] = 0.0
            ydot[-1] = 0.0
        return ydot

    for i in range(0, nt - 1):
        k1 = rhs(y[i, :], u[i, :])
        k2 = rhs(y[i, :] + 0.5 * model.h * k1, u[i, :])
        k3 = rhs(y[i, :] + 0.5 * model.h * k2, u[i, :])
        k4 = rhs(y[i, :] + model.h * k3, u[i, :])
        y[i + 1, :] = y[i, :] + model.h / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)

    # Observation
    # z = y[model.grid.iObs, :]
    z = observable(y, model)

    return y, z, t, model


def observable(y, model):
    return y[:, model.grid.iObs]
