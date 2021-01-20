# REQUIRES PACKAGES Numpy AND Scipy INSTALLED
import numpy as np
import scipy.integrate
import scipy.interpolate


class ddeVar:
    """ special function-like variables for the integration of DDEs """

    def __init__(self, g, tc=0):
        """ g(t) = expression of Y(t) for t<tc """

        self.g = g
        self.tc = tc
        # We must fill the interpolator with 2 points minimum
        self.itpr = scipy.interpolate.interp1d(
            np.array([tc - 1, tc]),  # X
            np.array([self.g(tc), self.g(tc)]).T,  # Y
            kind='linear', bounds_error=False,
            fill_value=self.g(tc))

    def update(self, t, Y):
        """ Add one new (ti,yi) to the interpolator """

        self.itpr.x = np.hstack([self.itpr.x, [t]])
        # Y2 = Y if (Y.size == 1) else np.array([Y]).T
        Y2 = np.array([Y]).T
        self.itpr.y = np.hstack([self.itpr.y, Y2])
        self.itpr.fill_value = Y
        self.itpr._y = self.itpr._reshape_yi(self.itpr.y)

    def __call__(self, t=0):
        """ Y(t) will return the instance's value at time t """

        return (self.g(t) if (t <= self.tc) else self.itpr(t))


class dde(scipy.integrate.ode):
    """ Overwrites a few functions of scipy.integrate.ode"""

    def __init__(self, f, jac=None):
        def f2(t, y, args):
            return f(self.Y, t, *args)

        scipy.integrate.ode.__init__(self, f2, jac)
        self.set_f_params(None)

    def integrate(self, t, step=0, relax=0):
        scipy.integrate.ode.integrate(self, t, step, relax)
        self.Y.update(self.t, self.y)
        return self.y

    def set_initial_value(self, Y):
        self.Y = Y  # !!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(self, Y(Y.tc), Y.tc)


def ddeint(func, g, tt, fargs=None):
    """
    Similar to scipy.integrate.odeint. Solves a Delay differential
    Equation system (DDE) defined by ``func`` with history function ``g``
    and potential additional arguments for the model, ``fargs``.
    Returns the values of the solution at the times given by the array ``tt``.

    Example:
    --------

    We will solve the delayed Lotka-Volterra system defined as

    For t < 0:
    x(t) = 1+t
    y(t) = 2-t

    For t > 0:
    dx/dt =  0.5* ( 1- y(t-d) )
    dy/dt = -0.5* ( 1- x(t-d) )

    Note that here the delay ``d`` is a tunable parameter of the model.

    ---
    import numpy as np
    def model(XY,t,d):
        x, y = XY(t)
        xd, yd = XY(t-d)
        return np.array([0.5*x*(1-yd), -0.5*y*(1-xd)])
    g = lambda t : np.array([1+t,2-t]) # 'history' at t<0
    tt = np.linspace(0,30,20000) # times for integration
    d = 0.5 # set parameter d 
    yy = ddeint(model,g,tt,fargs=(d,)) # solve the DDE !

    """

    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(g, tt[0]))
    dde_.set_f_params(fargs if fargs else [])
    return np.array([g(tt[0])] + [dde_.integrate(dde_.t + dt)
                                  for dt in np.diff(tt)])


def simulateModel(y0, t0, u, model):

    beta, gamma, eta = 2.0, 1.0, 9.65
    tau = model.params['tau']

    if 'SigZ' in model.params:
        SigZ = model.params['SigZ']
    else:
        SigZ = 0.0

    ny = len(y0)
    nt = u.shape[0]
    T = nt * model.h
    t = np.linspace(0, T, nt+1) + t0

    ty = np.linspace(t0 - tau, t0, ny)
    f = scipy.interpolate.interp1d(ty, y0, fill_value="extrapolate")

    if model.dimZ > 1:
        nLag_float = tau / model.h / (model.dimZ - 1)

    def iniVal(t_):
        return [f(t_)]

    def control(t_):
        return u[min(len(t[t <= t_]) - 1, nt-1)]

    def rhs(y_, t_):
        return [beta * y_(t_ - tau)[0] / (1.0 + np.power(y_(t_ - tau)[0], eta)) - gamma * y_(t_)[0] + control(t_)]
    # rhs = lambda y_, t_: beta * y_(t_ - tau) / (1.0 + np.power(y_(t_ - tau), eta)) - gamma * y_(t_) + control(t_)

    y = ddeint(rhs, iniVal, t)
    yAll = np.concatenate((y0, y[1:,0]), axis=0)
    z = np.zeros([nt, model.dimZ], dtype=float)
    for i in range(model.dimZ - 1):
        # z[:, i] = yAll[i * nLag:-(model.dimZ - i - 1) * nLag]
        z[:, -(i + 1)] = yAll[int(round(i * nLag_float)) + 1:-int(round((model.dimZ - i - 1) * nLag_float))] + \
                         SigZ * np.random.normal(loc=0.0, scale=np.sqrt(model.h), size=(z.shape[0]))
    # z[:, -1] = yAll[(model.dimZ - 1) * nLag:]
    z[:, 0] = yAll[int(tau / model.h) + 1:] + SigZ * np.random.normal(loc=0.0, scale=np.sqrt(model.h), size=(z.shape[0]))

    y = np.zeros([nt, ny], dtype=float)
    for i in range(ny - 1):
        y[:, i] = yAll[i + 1: -ny + i + 1]
    y[:, -1] = yAll[(ny - 1) + 1:]

    return y, z, t[1:], model
