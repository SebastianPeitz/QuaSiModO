import numpy as np


def simulateModel(y0, t0, u, model):

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

    # Distributed control input via characteristic function
    i11 = 0
    i12 = 19
    xSin = np.linspace(-0.5 * np.pi, 1.5 * np.pi, i12 - i11 + 1)
    Chi_u1 = np.zeros(nx, dtype=float)
    Chi_u1[i11: i12 + 1] = 0.5 * (np.sin(xSin) + 1.0)

    # Time integration
    for i in range(1, nt):
        y[i, :] = y[i - 1, :] - (model.h * y[i - 1, :] * (y[i - 1, :] - y[i - 1, im]) / model.grid.dx) + (
                viscosity * model.h * (y[i - 1, ip] - 2 * y[i - 1, :] + y[i - 1, im]) / (
                model.grid.dx * model.grid.dx)) + model.h * Chi_u1 * u[i, 0]

    # Observation
    # z = y[model.grid.iObs, :]
    z = observable(y, model)

    return y, z, t, model


def observable(y, model):
    return y[:, model.grid.iObs]
