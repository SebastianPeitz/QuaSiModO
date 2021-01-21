from helpers import hypercube
from scipy import interpolate
from scipy import sparse
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.io import savemat
from sys import exit
import importlib
import itertools
import numpy as np
import pickle


class ClassModel:
    """ClassModel

    This class contains
    - handle to the model
    - call to the model integrator
    - routine for computing observables in space (xObs)
    - parameters for the numerical solution (initial time, lag time)
    - function to construct a control grid

    Input
    1) model in form of
       - py-file placed in "models"
       - ClassOpenFOAM object
    2) h: step length
    3) Reynolds number: inverse of the viscosity
    4) iObs: Indices of states to observe in a finite-dimensional system (if None, then all are used)
    5) y0 (default = None): Initial condition of the full state
    6) uMin: array of lower bounds for the control input
    7) uMax: array of upper bounds for the control input
    8) dimZ: dimension of the observable
    9) typeUGrid: type of control grid that is created from uMin and uMax in the routine createControlGrid
    10) nGridU: number of sections the control input is divided into between uMin and uMax (component-wise)
    11) uGrid: pass the grid of controls directly to the model
    12) writeY: flag whether the full state should be written or not

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05/2020
    """

    y0 = []  # initial condition of full state
    grid = []  # container for a numerical grid, if required
    h = []  # step length of the integrator or - when using an external integrator - the step length in the time series
    Re = []  # Reynolds number (used as the inverse viscosity in many PDE problems)
    iObs = []  # array of indices of grid points at which the full state y is observed: z = y[iObs]

    dimZ = []
    dimU = []

    uGrid = []
    uC = []
    nU = []
    typeUGrid = []

    def __init__(self, modelFileOrClass, uMin, uMax, h, dimZ, params=None, iObs=None, y0=None,
                 typeUGrid=None, nGridU=1, uGrid=None, writeY=True, SigY=None, SigZ=None):

        print('Creating ClassModel with uMin = ' + str(uMin) + '; uMax = ' + str(uMax) + '; h = ' + str(h) +
              '; dimZ = ' + str(dimZ) + '; iObs = ' + str(iObs) + '; y0 = ' + str(y0) +
              '; typeUGrid = ' + str(typeUGrid) + '; nGridU = ' + str(nGridU))

        self.dimZ = dimZ

        self.h = h

        self.uMin = np.array(uMin)
        self.uMax = np.array(uMax)
        self.dimU = len(uMin)

        if uGrid is not None:
            self.nU = uGrid.shape[0]
            self.uC = uGrid[0]
            self.uGrid = uGrid
        else:
            if typeUGrid is not None:
                self.createControlGrid(self.uMin, self.uMax, typeUGrid, nGridU)

        if y0 is not None:
            self.y0 = y0

        if params is not None:
            self.params = params

        if iObs is not None:
            self.iObs = iObs

        if SigY is not None:
            self.SigY = SigY

        if SigZ is not None:
            self.SigZ = SigZ

        self.writeY = writeY

        # set the model structure depending on the input type in "modelFileOrClass"
        # self.model calls the integrator that yields the trajectories for y and z
        # self.observable is a function that -- given y and model -- yields z
        #
        # Type of "modelFileOrClass":
        # 1) the right-hand side of an ODE
        if callable(modelFileOrClass):
            self.rhs = modelFileOrClass
            if SigY is None:
                self.model = self.simulateODE
            else:
                self.model = self.simulateSDE

            if SigZ is None:
                self.observable = self.observeODE
            else:
                self.observable = self.observeSDE
            self.calcJ = None
        else:

            # 2) the name of a python file placed in the "models" folder
            if isinstance(modelFileOrClass, str):
                modelName = modelFileOrClass[:-3]

            # 3) an OpenFOAM class variable
            else:
                self.OF = modelFileOrClass
                modelName = self.OF.modelFile[:-3]

            moduleModel = importlib.import_module("models." + modelName, package="simulateModel")
            self.model = moduleModel.simulateModel

            if hasattr(moduleModel, 'observable'):
                self.observable = moduleModel.observable
            else:
                self.observable = None

            if hasattr(moduleModel, 'calcJ'):
                self.calcJ = moduleModel.calcJ
            else:
                self.calcJ = None

    # calls the model that has been set up in __init__
    def integrate(self, y0, u, t0):
        return self.model(y0, t0, u, self)

    # integrator
    def RK4(self, y0, t0, u):
        nt = u.shape[0]
        T = (nt - 1) * self.h
        t = np.linspace(0.0, T, nt) + t0
        y = np.empty([nt, len(y0)], dtype=float)
        y[0, :] = y0
        for i in range(nt - 1):
            k1 = self.rhs(y[i, :], u[i, :])
            k2 = self.rhs(y[i, :] + 0.5 * self.h * k1, u[i, :])
            k3 = self.rhs(y[i, :] + 0.5 * self.h * k2, u[i, :])
            k4 = self.rhs(y[i, :] + self.h * k3, u[i, :])
            y[i + 1, :] = y[i, :] + self.h / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)

        return y, t

    def EulerMaruyama(self, y0, t0, u):
        nt = u.shape[0]
        T = (nt - 1) * self.h
        t = np.linspace(0.0, T, nt) + t0
        y = np.empty([nt, len(y0)], dtype=float)
        y[0, :] = y0
        for i in range(nt - 1):
            y[i + 1, :] = y[i, :] + self.h * self.rhs(y[i, :], u[i, :]) + \
                          self.SigY * np.random.normal(loc=0.0, scale=np.sqrt(self.h), size=(1, len(y0)))

        return y, t

    # simulation model for ODEs
    def simulateODE(self, y0, t0, u, model):
        # Solve
        y, t = self.RK4(y0, t0, u)

        # Observation
        z = self.observable(y)

        return y, z, t, model

    # simulation model for SDEs
    def simulateSDE(self, y0, t0, u, model):
        # Solve
        y, t = self.EulerMaruyama(y0, t0, u)

        # Observation
        z = self.observable(y)

        return y, z, t, model

    # observable for ODEs
    def observeODE(self, y):
        if len(self.iObs) > 0:
            z = y[:, self.iObs]
        else:
            z = y

        return z

    # observable for ODEs
    def observeSDE(self, y):

        if len(self.iObs) > 0:
            nObs = len(self.iObs)
        else:
            nObs = y.shape[1]

        z = np.zeros([y.shape[0], nObs], dtype=float)

        if len(self.iObs) > 0:
            for i in range(y.shape[0]):
                z[i, :] = y[i, self.iObs] + np.sqrt(self.SigZ) * np.random.normal(size=(1, nObs))
        else:
            for i in range(y.shape[0]):
                z[i, :] = y[i, :] + self.SigZ * np.random.normal(loc=0.0, scale=np.sqrt(self.h), size=(1, nObs))

        return z

    # computes a 1D grid using the domain length L and a step size dx
    def setGrid1D(self, L, dx, xObs=None):
        self.grid = Class1DGrid(L, dx, xObs)

    def createControlGrid(self, uMin=None, uMax=None, typeUGrid='cube', nGridU=1):
        """
        Creates a grid (uGrid) of controls for which the system can be evaluated, depending on typeUGrid.
        The different controls are accessed via the first index, the second index then provides access to the individual
        entries of the respective control

        typeUGrid =
        1) 'cube': Hypercube with uMin and uMax as corners
        2) 'cubeCenter': Hypercube with uMin and uMax as corners and the center uC = 0.5 * (uMin + uMax) in addition
        3) 'centerStar': Finite-difference-like star of controls, including the center uC = 0.5 * (uMin + uMax)
        4) 'oneSidedStar': Half of a finite-difference-like star of controls, including the center uC = uMin

        nGridU determines the number of segments along each dimension where control grid points are placed (default = 1)
        """

        print('Creating control grid with uMin = ' + str(uMin) + '; uMax = ' + str(uMax) + '; typeUGrid = ' +
              str(typeUGrid) + '; nGridU = ' + str(nGridU))

        if uMin is None:
            uMin = self.uMin

        if uMax is None:
            uMax = self.uMax

        self.typeUGrid = typeUGrid

        if (typeUGrid == 'cube') or (typeUGrid == 'cubeCenter'):
            self.uC = 0.5 * (uMin + uMax)
            self.nU = (1 + nGridU) ** self.dimU
            uGrid = np.fliplr(np.array(
                hypercube.hypercube_grid_points(self.dimU, self.nU, (1 + nGridU) * np.ones(self.dimU, dtype=int),
                                                np.flip(uMin),
                                                np.flip(uMax), np.ones(self.dimU, dtype=int))).T)
            for i in range(self.dimU):
                uGrid[:, i] = uGrid[:, i] + uMin[i] - uGrid[0, i]  # 0.5 * (uMax[i] - uMin[i])

            if typeUGrid == 'cubeCenter':
                uGrid2 = np.zeros([uGrid.shape[0] + 1, uGrid.shape[1]], dtype=float)
                uGrid2[:-1, :] = uGrid
                uGrid2[-1, :] = self.uC
                uGrid = uGrid2

        elif typeUGrid == 'centerStar':
            self.uC = 0.5 * (uMin + uMax)
            self.nU = 2 * self.dimU + 1
            uGrid = np.zeros([self.nU, self.dimU], dtype=float)
            uGrid[0, :] = self.uC
            for i in range(self.dimU):
                uGrid[2 * i + 1, :] = self.uC
                uGrid[2 * i + 1, i] = uMin[i]
                uGrid[2 * i + 2, :] = self.uC
                uGrid[2 * i + 2, i] = uMax[i]

        elif typeUGrid == 'oneSidedStar':
            self.uC = uMin
            self.nU = self.dimU + 1
            uGrid = np.zeros([self.nU, self.dimU], dtype=float)
            uGrid[0, :] = self.uC
            for i in range(self.dimU):
                uGrid[i + 1, :] = uMin
                uGrid[i + 1, i] = uMax[i]

        else:
            print('Error in "ClassModel.__init__.createControlGrid": Please specify "typeUGrid" as "cube", "centerStar"'
                  ' or "oneSidedStar"')
            exit(1)

        self.uGrid = uGrid
        return uGrid

    def mapAlphaToU(self, alpha):
        u = np.zeros([alpha.shape[0], self.dimU], dtype=float)
        for i in range(alpha.shape[0]):
            for iu in range(self.nU - 1):
                u[i, :] += alpha[i, iu] * self.uGrid[iu, :]
            u[i, :] += (1.0 - np.sum(alpha[i, :])) * self.uGrid[-1, :]
        return u


class Class1DGrid:
    """Class1DGrid

    This class creates a 1D grid and potentially the indices of observed grid points

    Input
    1) L = length of domain
    2) dx = grid size
    3) xObs = array of positions where the state is observed

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05/2020
    """

    xObs = []

    def __init__(self, L, dx, xObs=None):

        print('Creating 1D grid with L = ' + str(L) + '; dx = ' + str(dx) + '; xObs = ' + str(xObs))

        self.L = L
        self.dx = dx
        self.x = np.arange(0, L + dx, dx)
        if xObs is not None:
            self.xObs = xObs
            self.nz = len(xObs)
            self.iObs = np.zeros(len(xObs), dtype=int)
            xTmp = np.zeros(len(self.x), dtype=float)
            for i in range(len(xObs)):
                xTmp[:] = xObs[i]
                self.iObs[i] = int(np.argmin(np.absolute(self.x - xTmp)))


class ClassControlDataSet:
    """ClassControlDataSet

    This class contains routines for creating control sequences for data sampling and for the construction of raw data
    sets which can be used for the surrogate modeling later

    Inputs (optional, can also be specified in createControlSequence or createData, respectively)
    1) h: time step for the control sequence. u is constant over each time step
    2) T: final time of control sequence
    3) y0: a (set of) initial condition(s) for which the simulation is performed

    Output
    1) Control trajectory

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05/2020
    """

    t = []
    T = []
    rawData = []
    nDelay = []
    dimZ = []

    def __init__(self, h=None, T=None, y0=None):
        """
        Creates the time step array and set y0 if given
        """

        print('Creating ClassControlDataSet with h = ' + str(h) + '; T = ' + str(T) + '; y0 = ' + str(y0))

        if h is not None and T is not None:
            self.h = h
            self.T = T
            self.t = np.linspace(0, T, int(T / h + 1))

        if y0 is not None:
            self.y0 = np.array(y0)

    def createControlSequence(self, model, h=None, T=None, uGrid=None, nhMin=1, nhMax=10,
                              typeSequence='piecewiseConstant', u=None, iu=None, periodicParameters=None):
        """
        Creates a control sequence based on the sequence type and the uGrid specified in 'ClassModel.createControlGrid'
        via the variable typeUGrid. The structure of the output is a 3-dimensional array, where the third index
        indicates the number of the control trajectory that needs to be simulated

        Inputs:
        1) model: ClassModel type model
        2) h: time step for the control sequence. u is constant over each time step
        3) T: final time of control sequence
        4) uGrid: array of inputs. The different controls are indexed via the first index
        5) nhMin: minimal number of time steps over which the control is constant
        6) nhMax: maximal number of time steps over which the control is constant
        7) typeSequence =
            a) 'piecewiseConstant': piecewise constant control, selected randomly from self.uGrid.
                                    Random length of intervals bounded by nhMin and nhMax.
            b) 'piecewiseLinear': piecewise linear interpolation between inputs selected randomly from self.uGrid.
                                  Random length of intervals bounded by nhMin and nhMax.
            c) 'spline': Spline interpolation between inputs selected randomly from self.uGrid.
                         Random length of intervals bounded by nhMin and nhMax.
            d) Numerical value: Constant control input over the entire trajectory
            e) 'sine': Sine function u(t) = a * sin(2 * pi * b * (t - c)). In that case, the input
                       periodicParameters = [a, b, c] needs to be specified
        8) u: Previously created sequence

        9) iu: index of controls to input 8)
        10) periodicParameters = [a, b, c]: parameter in the case of sine input u(t) = a * sin(2 * pi * b * (t - c))
        """

        if T is None:
            T = self.T

        if h is None:
            h = self.h

        nt = int(T / h + 1)
        self.t = np.linspace(0, T, nt)

        if uGrid is None:
            if len(model.uGrid) == 0:
                print('Error in "ClassControlDataSet.createControlSequence": Please specify control grid first via '
                      '"createControlGrid" or provide variable "uGrid" directly')
                exit(1)
            else:
                uGrid = model.uGrid

        print('Creating control sequence with h = ' + str(h) + '; T = ' + str(T) + '; uGrid = ' + str(uGrid) +
              '; nhMin = ' + str(nhMin) + '; nhMax = ' + str(nhMax) + '; typeSequence = ' + str(typeSequence))

        if u is None:
            if iu is not None:
                print('Error in "ClassControlDataSet.createControlSequence": Please specify either both u and iu or '
                      'none.')
                exit(1)
            else:
                u = list()
                iu = list()
        else:
            if iu is None:
                print('Error in "ClassControlDataSet.createControlSequence": Please specify either both u and iu or '
                      'none.')
                exit(1)

        u_ = np.zeros([nt, model.dimU], dtype=float)
        iu_ = np.zeros([nt, 1], dtype=int)

        if isinstance(typeSequence, str):
            if typeSequence == 'piecewiseConstant':
                i1, i2 = 0, 0
                while i2 < nt:
                    i2 = np.minimum(i1 + np.random.randint(nhMin, nhMax + 1), nt)
                    iu1 = np.random.randint(model.nU)
                    for j in range(i1, i2):
                        iu_[j, 0] = iu1
                        u_[j, :] = uGrid[iu1, :]

                    i1 = i2

            elif typeSequence == 'piecewiseLinear':
                i1, i2 = 0, 0
                iu1 = 0
                iu_[i1, 0] = iu1
                u_[i1, :] = uGrid[iu1, :]
                while i2 < nt:
                    i2 = np.minimum(i1 + np.random.randint(nhMin, nhMax + 1), nt)
                    iu2 = np.random.randint(model.nU)
                    if i2 - i1 < 2:
                        u_[i1, :] = uGrid[iu1, :]
                    else:
                        for j in range(i1, i2):
                            iu_[j, 0] = iu2
                            u_[j, :] = uGrid[iu1, :] + (uGrid[iu2, :] - uGrid[iu1, :]) * (j - i1) / (i2 - i1 - 1)

                    i1 = i2

            elif typeSequence == 'spline':
                i1, i2 = 0, 0
                iSupport = np.zeros(nt, dtype=int)
                uSupport = np.zeros([nt, model.dimU], dtype=float)
                iSupport[0] = 0
                uSupport[0, :] = uGrid[0, :]
                s = 0
                while i2 < nt - 1:
                    s += 1
                    i2 = np.minimum(i1 + np.random.randint(nhMin, nhMax + 1), nt - 1)
                    iSupport[s] = i2
                    uSupport[s, :] = uGrid[np.random.randint(model.nU), :]
                    i1 = i2

                iSupport = iSupport[:s + 1]
                uSupport = uSupport[:s + 1, :]

                for i in range(model.dimU):
                    tck = interpolate.splrep(self.t[iSupport], uSupport[:, i], s=0)
                    u_[:, i] = interpolate.splev(self.t, tck, der=0)

                iu_ = []

            elif typeSequence == 'sine':
                if periodicParameters is None:
                    print('Error in "ClassControlDataSet.createControlSequence": Please specify '
                          'typeSequence = [a, b, c] if you set typeSequence to "sine".')
                    exit(1)

                u_[:, 0] = periodicParameters[0] * np.sin(2 * np.pi * periodicParameters[1] * (self.t - periodicParameters[2]))

        else:
            uConst = np.zeros([1, model.dimU], dtype=float)
            uConst[0, :] = typeSequence
            iu_[:] = mapUToIu(uConst, model.uGrid)
            u_ = mapIuToU(iu_, model.uGrid)

        u.append(u_)
        iu.append(iu_)
        return u, iu

    def createData(self, model=None, y0=None, u=None, savePath=None, loadPath=None):
        """
        Creates the corresponding data structure, depending on the model reduction method.
        If simulate=True, then the data is created by simulating the model.
        Saving and loading the raw data can be realized by providing the corresponding paths (savePath or loadPath)

        Inputs:
        1) model: ClassModel type model
        2) y0: a (set of) initial condition(s) for which the simulation is performed
        3) u: a (set of) control(s) for which the simulation is performed
        4) savePath: path + filename to which the raw data should be stored
        5) loadPath: path + filename from which the raw data should be obtained. If specified, the data is just loaded
                     and the routine ended prematurely

        Output (also stored directly in self.rawData)
        1) rawData structure consisting of the variables (y, z, t, u, iu)
        """

        print('Creating rawData by simulation with savePath = ' + str(savePath) + '; loadPath = ' + str(loadPath))

        if loadPath is not None:
            self.rawData = ClassRawData(loadPath=loadPath)
            return

        if model is None:
            print('Error in "ClassControlDataSet.createData": Please specify model for simulation')
            exit(1)

        if u is None:
            print('Error in "ClassControlDataSet.createData": Please specify control u')
            exit(1)

        if y0 is None:
            y0 = list()
            y0.append(model.y0)
        else:
            if isinstance(y0[0], float):
                y0_ = list()
                y0_.append(y0)
                y0 = y0_

        nIC = len(y0)
        nU = len(u)

        yAll, zAll, tAll, uAll, iuAll = list(), list(), list(), list(), list()
        for i in range(nU):
            for j in range(nIC):
                [y, z, t, _] = model.integrate(y0[j], u[i], 0.0)
                yAll.append(y)
                zAll.append(z)
                tAll.append(t)
                uAll.append(u[i])
                iuAll.append(mapUToIu(u[i], model.uGrid))

        self.rawData = ClassRawData(y=yAll, z=zAll, t=tAll, u=uAll, iu=iuAll)

        if savePath is not None:
            self.rawData.save(savePath=savePath)

    def prepareData(self, model, method='None', rawData=None, nLag=1, nDelay=0, typeDer='FD'):
        """
        Takes in the rawData (either passed directly or stored in the class) and creates structured data according to
        the selected method. For each of the (model.nU) entries in the control grid, a list entry is created:
        X[i] corresponds to the input uGrid[i, :] for i in {0, 1, ... model.nU - 1}

        Inputs
        1) model (type ClassModel)
        2) method (type string): Depending on which entries the string contains, different types of output data are
           created:
           a) 'Y': X shifted forward in time by nLag entries
           b) 'dX': Calculates the derivative of X (depending on typeDer, either using the model or finite differences)
           c) 'XU': Creates the product of states times controls (e.g., for Koopman generator models with inputs)
           d) 'YU': Creates the time shift of XU by nLag entries (e.g., for a Koopman operator approximation for c))
           e) 'dXU': Creates the time derivative of 'XU'
        3) rawData (type ClassRawData): Collected data. Can alternatively be accessed from self.rawData
        4) nLag (type int): Number of time steps the time-shifted data matrices are shifted forward in time
        5) nDelay (type int, default = 0): Number of time delayed observations
        6) typeDer (type string): Method for the numerical computation of derivatives ('FD': forward differences or
           'CD': central differences). If the model possesses a right-hand side description (model.rhs), then the
           derivative is computed by evaluating the model)

        Output
        1) data structure containing the prepared data matrices ({'X': X, 'Y': Y, 'dX': dX, 'XU': XU, 'dXU': dXU})
        """

        if rawData is None:
            rawData = self.rawData

        self.dimZ = rawData.z[0].shape[1]
        self.nDelay = nDelay

        print('Creating ClassModel with method = ' + str(method) + '; rawData = ' + str(rawData) + '; nLag = ' +
              str(nLag) + '; nDelay = ' + str(nDelay) + '; typeDer = ' + str(typeDer))

        flagY = (str.find(method, 'Y') >= 0)
        flagDX = (str.find(method, 'dX') >= 0)
        flagXU = (str.find(method, 'XU') >= 0)
        flagYU = (str.find(method, 'YU') >= 0)
        flagDXU = (str.find(method, 'dXU') >= 0)

        X, Y, dX, XU, YU, dXU, u = list(), list(), list(), list(), list(), list(), list()

        for i in range(model.nU):
            X.append(list())
            Y.append(list())
            dX.append(list())
            XU.append(list())
            YU.append(list())
            dXU.append(list())
            u.append(model.uGrid[i, :])

        if flagDX or flagDXU:
            if hasattr(model, 'rhs'):
                dy = list()
                dz = list()
                for i in range(len(rawData.z)):
                    dy.append(calcDerivative(rawData.y[i], model, typeDer, U=rawData.u[i]))
                    dz.append(model.observable(dy[-1]))
            else:
                dz = list()
                for i in range(len(rawData.z)):
                    dz.append(calcDerivative(rawData.z[i], model, typeDer))

        for i in range(len(rawData.z)):
            if flagY or flagYU:
                for j in range(rawData.z[i].shape[0] - (self.nDelay + 1) * nLag):
                    if self.nDelay == 0:
                        if not any(rawData.iu[i][j: j + nLag, 0] != rawData.iu[i][j, 0]):
                            X[rawData.iu[i][j, 0]].append(rawData.z[i][j, :])
                            Y[rawData.iu[i][j, 0]].append(rawData.z[i][j + nLag, :])
                            if flagDX:
                                dX[rawData.iu[i][j, 0]].append(dz[i][j, :])
                            if flagXU:
                                pass
                            if flagYU:
                                pass
                            if flagDXU:
                                pass
                    else:
                        if not any(rawData.iu[i][j + nDelay * nLag: j + (1 + nDelay) * nLag, 0] != rawData.iu[i][j + nDelay * nLag, 0]):
                            X[rawData.iu[i][j + nDelay * nLag, 0]].append(self.stackZ(rawData.z[i][j: j + self.nDelay * nLag + 1, :], nLag))
                            Y[rawData.iu[i][j + nDelay * nLag, 0]].append(self.stackZ(rawData.z[i][j + nLag: j + (self.nDelay + 1) * nLag + 1, :], nLag))
                        
            else:
                for j in range(rawData.z[i].shape[0]):
                    X[rawData.iu[i][j, 0]].append(rawData.z[i][j, :])
                    if flagDX:
                        dX[rawData.iu[i][j, 0]].append(dz[i][j, :])
                    if flagXU:
                        pass
                    if flagYU:
                        pass
                    if flagDXU:
                        pass

        for i in range(model.nU):
            X[i] = np.array(X[i])
            Y[i] = np.array(Y[i])
            dX[i] = np.array(dX[i])
            XU[i] = np.array(XU[i])
            YU[i] = np.array(YU[i])
            dXU[i] = np.array(dXU[i])

        data = {'X': X, 'Y': Y, 'dX': dX, 'XU': XU, 'dXU': dXU, 'u': u}

        return data

    def stackZ(self, z, hShM):
        zOut = np.zeros([self.dimZ * (1 + self.nDelay)], dtype=float)
        for i in range(self.nDelay + 1):
            zOut[i * self.dimZ: (i + 1) * self.dimZ] = z[-(i * hShM + 1), :]
        return zOut


def calcDerivative(X, model, typeDer=None, U=None):
    dX = np.zeros(X.shape, dtype=float)
    if hasattr(model, 'rhs') and (U is not None):
        for i in range(X.shape[0]):
            dX[i, :] = model.rhs(X[i, :], U[i, :])
    else:
        if typeDer is None:
            typeDer = 'FD'

        if typeDer == 'CD':
            dX[0, :] = (X[1, :] - X[0, :]) / model.h
            dX[-1, :] = (X[-1, :] - X[-2, :]) / model.h
            for i in range(1, X.shape[0] - 1):
                dX[i, :] = (X[i + 1, :] - X[i - 1, :]) / (2.0 * model.h)
        elif typeDer == 'FD':
            dX[-1, :] = (X[-1, :] - X[-2, :]) / model.h
            for i in range(X.shape[0] - 1):
                dX[i, :] = (X[i + 1, :] - X[i, :]) / model.h

    return dX


class ClassRawData:
    """ClassRawData

    This class contains raw simulation data and functions for saving and loading the data

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05/2020
    """
    y, z, t, u, iu = [], [], [], [], []

    def __init__(self, y=None, z=None, t=None, u=None, iu=None, savePath=None, loadPath=None):

        if loadPath is not None:
            self.load(loadPath)
            return

        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        if t is not None:
            self.t = t
        if u is not None:
            self.u = u
        if iu is not None:
            self.iu = iu

        if savePath is not None:
            self.save(savePath)

    def save(self, savePath):
        np.savez(savePath, y=self.y, z=self.z, t=self.t, u=self.u, iu=self.iu, allow_pickle=True)

    def load(self, loadPath):
        if loadPath[-4:] != '.npz':
            loadPath = loadPath + '.npz'
        dataIn = np.load(loadPath, allow_pickle=True)

        self.y = arrayToList(dataIn['y'])
        self.z = arrayToList(dataIn['z'])
        self.t = arrayToList(dataIn['t'])
        self.u = arrayToList(dataIn['u'])
        self.iu = arrayToList(dataIn['iu'])


def arrayToList(x):
    if type(x) != type(list()):
        y = list()
        if x[0,...].shape == ():
            for i in range(x.shape[0]):
                y.append(x[i, ...].item())
        else:
            for i in range(x.shape[0]):
                y.append(x[i, ...])
        
        return y
    else:
        return x


class ClassModelData(object):
    pass


class ClassSurrogateModel:
    """ClassSurrogateModel

    This class contains functions for the preparation of data, the integration of the relaxed surrogate model and
    some additional help functions. The "surrogateModelFile" should contain the following functions:
    1) timeTMap(z0, t0, iu, surrogateModel) -> z, t, surrogateModel
       Discrete mapping over one time step surrogateModel.h using the i-th input (contained in iu). The time step and
       the control grid are stored in surrogateModel.h and surrogateModel.uGrid, respectively. All additional data
       required for the surrogate model (including memory terms for the next time step) can be accessed via
       surrogateModel.modelData, which is first created in the function "createSurrogateModel"
    2) createSurrogateModel(modelData, X, Y) -> modelData
       Creates a data-driven surrogate model using X and Y. Additional options can be stored in modelData
    3) updateSurrogateModel(modelData, X, Y) -> modelData

    Inputs
    1) surrogateModelFile: name of the .py file stored in surrogateModels
    2) uGrid: grid of fixed controls
    3) h: step size of the surrogate model
    4) z0: initial condition (can also be left blank if specified later in the MPC class)
    5) dimZ: dimension of the observable z
    6) nDelay (type int, default = 0): Number of time delayed observations
    7) Arbitrary additional parameters required for the surrogate model: These are stored as a dictionary in .modelData

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05/2020
    """

    z0 = None
    dimZ = None
    modelData = ClassModelData()

    integrate = None
    timeTMap = None
    createSurrogateModel = None
    saveSurrogateModel = None
    updateSurrogateModel = None
    calcJ = None

    uGrid = None
    nU = None
    dimU = None

    hShM = 1

    X, Y = [], []

    def __init__(self, surrogateModelFile, uGrid, h, z0=None, dimZ=None, nDelay=0, **kwargs):

        stringOut = 'Creating ClassSurrogateModel with surrogateModelFile = ' + str(surrogateModelFile) + '; uGrid = ' \
                    + str(uGrid) + '; h = ' + str(h) + '; z0 = ' + str(z0) + '; dimZ = ' + str(dimZ)

        if len(kwargs) > 0:
            stringOut += ' and free parameter(s) '
        for key, value in kwargs.items():
            stringOut += str(key) + ' = ' + str(value) + ';'
        print(stringOut[:-1])

        # Set parameters that have been passed to the class
        if z0 is not None:
            self.z0 = np.array(z0)
            if dimZ is None:
                self.dimZ = len(z0)

        if dimZ is not None:
            self.dimZ = dimZ

        self.nDelay = nDelay
        self.h = h

        if uGrid is None:
            print('Error in "ClassSurrogateModel.__init__": please provide uGrid')
            exit(1)
        else:
            self.uGrid = uGrid
            self.nU = uGrid.shape[0]
            self.dimU = uGrid.shape[1]

        if surrogateModelFile[-3:] == '.py':
            surrogateModelFile = surrogateModelFile[:-3]
        moduleSurrogateModel = importlib.import_module("surrogateModels." + surrogateModelFile)
        if hasattr(moduleSurrogateModel, 'timeTMap'):
            self.timeTMap = moduleSurrogateModel.timeTMap
        if hasattr(moduleSurrogateModel, 'createSurrogateModel'):
            self.createSurrogateModel = moduleSurrogateModel.createSurrogateModel
        if hasattr(moduleSurrogateModel, 'updateSurrogateModel'):
            self.updateSurrogateModel = moduleSurrogateModel.updateSurrogateModel
        if hasattr(moduleSurrogateModel, 'calcJ'):
            self.calcJ = moduleSurrogateModel.calcJ
        if hasattr(moduleSurrogateModel, 'saveSurrogateModel'):
            self.saveSurrogateModel = moduleSurrogateModel.saveSurrogateModel
        if hasattr(moduleSurrogateModel, 'loadSurrogateModel'):
            self.loadSurrogateModel = moduleSurrogateModel.loadSurrogateModel

        setattr(self.modelData, 'h', self.h)
        setattr(self.modelData, 'uGrid', self.uGrid)
        setattr(self.modelData, 'nDelay', self.nDelay)
        for key, value in kwargs.items():
            setattr(self.modelData, key, value)

    def createROM(self, data, savePath=None, loadPath=None):

        if loadPath is not None:
            fIn = open(loadPath + '.pkl', 'rb')
            selfLoad = pickle.load(fIn)
            self.X = selfLoad.X
            self.Y = selfLoad.Y
            self.dimU = selfLoad.dimU
            self.dimZ = selfLoad.dimZ
            self.h = selfLoad.h
            self.hShM = selfLoad.hShM
            self.integrate = selfLoad.integrate
            self.nDelay = selfLoad.nDelay
            self.nU = selfLoad.nU
            self.uGrid = selfLoad.uGrid
            self.z0 = selfLoad.z0
            if self.loadSurrogateModel is not None:
                self.modelData = self.loadSurrogateModel(loadPath)
            else:
                setattr(self, 'modelData', selfLoad.modelData)
            return

        if self.createSurrogateModel is not None:
            self.modelData = self.createSurrogateModel(self.modelData, data)
            if savePath is not None:
                if self.saveSurrogateModel is not None:
                    self.saveSurrogateModel(self.modelData, savePath)
                    modelData_temp = self.modelData
                    self.modelData = []
                with open(savePath + '.pkl', 'wb') as fOut:
                    pickle.dump(self, fOut)
                if self.saveSurrogateModel is not None:
                    self.modelData = modelData_temp

        else:
            print('Error in "ClassSurrogateModel.createROM": No function "createSurrogateModel(modelData, data)" '
                  'defined in the .py file containing the model')
            exit(1)

    def updateROM(self, data):

        if self.updateSurrogateModel is not None:
            self.modelData = self.updateSurrogateModel(self.modelData, data)
        else:
            print('Error in "ClassSurrogateModel.updateROM": No function "updateSurrogateModel(modelData, data)" '
                  'defined in the .py file containing the model')
            exit(1)

    def integrateRelaxedTimeTMap(self, z0, t0, alpha):
        z = np.zeros([alpha.shape[0], self.dimZ * (1 + self.nDelay)], dtype=float)
        z[0, :] = z0
        zPlus = np.zeros([self.dimZ * (1 + self.nDelay)], dtype=float)
        time = t0
        for i in range(alpha.shape[0] - 1):
            zPlus[:] = 0.0
            for iu in range(self.nU - 1):
                ziu, _, _ = self.timeTMap(z[i, :], time, iu, self.modelData)
                zPlus += alpha[i, iu] * ziu
            ziu, _, _ = self.timeTMap(z[i, :], time, self.nU - 1, self.modelData)
            zPlus += (1.0 - np.sum(alpha[i, :])) * ziu
            z[i + 1, :] = zPlus
            time = time + self.h

        return z

    def integrateDiscreteInput(self, z0, t0, iu):
        z = np.zeros([iu.shape[0], self.dimZ * (1 + self.nDelay)], dtype=float)
        t = np.linspace(0.0, (iu.shape[0] - 1) * self.h, iu.shape[0]) + t0
        z[0, :] = z0
        ti = t0
        for i in range(iu.shape[0] - 1):
            z[i + 1, :], ti, self.modelData = self.timeTMap(z[i, :], ti, iu[i, 0], self.modelData)

        return z, t

    def mapAlphaToU(self, alpha):
        u = np.zeros([alpha.shape[0], self.dimU], dtype=float)
        for i in range(alpha.shape[0]):
            for iu in range(self.nU - 1):
                u[i, :] += alpha[i, iu] * self.uGrid[iu, :]
            u[i, :] += (1.0 - np.sum(alpha[i, :])) * self.uGrid[-1, :]
        return u


class ClassReferenceTrajectory:
    """ClassReferenceTrajectory

    This class contains the reference trajectory for the MPC problem as well as methods for its computation

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05/2020
    """

    zSurrogate = []
    iRefSurrogate = []

    def __init__(self, model, zRef, T=None, iRef=None):

        print('Creating ClassReferenceTrajectory with T = ' + str(T) + '; iRef = ' +
              str(iRef))

        zRef = np.asarray(zRef)

        if model.dimZ is None:
            print('Error in "ClassReferenceTrajectory.__init__": Please specify dimZ in ClassModel file')
            exit(1)

        if T is None:
            self.T = int((zRef.shape[0] - 1) / model.h)
        else:
            self.T = T

        if iRef is None:
            self.iRef = np.array(range(model.dimZ))
        else:
            self.iRef = np.asarray(iRef)

        n = int(self.T / model.h) + 1

        self.z = np.zeros([n, model.dimZ], dtype=float)

        if zRef.ndim == 1:
            for i in range(zRef.shape[0]):
                self.z[:, self.iRef[i]] = zRef[i]
        else:
            if zRef.shape[0] == 1:
                for i in range(zRef.shape[1]):
                    self.z[:, self.iRef[i]] = zRef[0, i]
            elif zRef.shape[0] != n:
                print('Error in "ClassReferenceTrajectory".__init__": number of time steps of zRef has to be either 1 '
                      'or "T / model.h + 1"')
                exit(1)
            else:
                if zRef.shape[1] == 1:
                    for i in range(len(self.iRef)):
                        self.z[:, self.iRef[i]] = zRef[:, 0]
                else:
                    for i in range(len(self.iRef)):
                        self.z[:, self.iRef[i]] = zRef[:, i]

        # self.z = sparse.csr_matrix(self.z, shape=[n, model.dimZ])


class ClassMPC:
    """ClassMPC

    This class contains the Model Predictive Control functionality. If "surrogateModel" is provided, then the MPC is
    performed using the surrogate model

    Inputs
    1) T: Length of the entire control horizon
    2) typeOpt (default = 'SUR'):
    3) np: prediction horizon length as multiples of model.h
    4) nc: control horizon length as multiples of model.h
    5) nch (default = 1): number of repeated applications of one input
    6) scipyMinimizeMethod (default = 'SLSQP'): optimization algorithm to be used by scipy.minimize
    7) scipyMinimizeOptions (default = {'ftol': 1e-08}): options for the algorithm executed in scipy.minimize

    Depending on typeOpt, the optimization problem is solved in different ways:
    1) 'combinatorial': Full evaluation of all possible combinations, grows exponentially with prediction horizon length
                        #Eval = ClassControlDataSet.nU ^ np
    2) 'SUR': Relaxation of the MIOCP and sum-up rounding to preserve integer controls (i.e., points on the uGrid)
    3) 'SUR_coarse': Relaxation of the MIOCP and sum-up rounding to preserve integer controls as 'SUR', but the input
       is constant over on control horizon length (i.e., points on the uGrid)
    4) 'continuous': Relaxation as in 2), but without the sum-up rounding

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05/2020
    """

    np = []
    nc = []
    bounds = []
    linConstraints = []
    dimZ = []

    Q, R, S, L = [], [], [], []

    def __init__(self, T=0.0, typeOpt='combinatorial', np=None, nc=None, nch=1,
                 scipyMinimizeMethod='SLSQP', scipyMinimizeOptions=None):

        print('Creating ClassMPC with T = ' + str(T) + '; typeOpt = ' + str(typeOpt) + '; np = ' + str(np) +
              '; nc = ' + str(nc) + '; nch = ' + str(nch) + '; scipyMinimizeMethod = ' +
              str(scipyMinimizeMethod) + '; scipyMinimizeOptions = ' + str(scipyMinimizeOptions))

        self.T = T
        self.typeOpt = typeOpt
        self.nch = nch

        if np is not None:
            self.np = np

        if nc is not None:
            self.nc = nc

        self.scipyMinimizeMethod = scipyMinimizeMethod
        self.scipyMinimizeOptions = scipyMinimizeOptions

        self.calcJ = self.calcJInternal

    def run(self, model, reference, surrogateModel=None, y0=None, z0=None, T=None, Q=[1.0], L=[0.0], R=[0.0], S=[0.0],
            savePath=None, updateSurrogate=False, iuInit=0):
        """ClassMPC.run

        This function solves the MPC problem over the time interval [0, T] by sequentially solving the following
        optimal control problem over the prediction horizon np and applying nc entries to the plant:

        min_{u in bounds} \int_{0}^{np * h) (z-zRef)^T * Q * (z-zRef) + u^T * R * u + \dot{u}^T * S * \dot{u} dt

        Depending on whether a surrogate model is supplied or not, the time step h corresponds to model.h or
        surrogateModel.h. \dot{u} is approximated using forward differences

        Inputs
        1) model (tytpe ClassModel)
        2) reference (tytpe ClassReferenceTrajectory)
        3) surrogateModel (tytpe ClassSurrogateModel, default = None): If provided, then the MPC problem is solved using
           the surrogate model
        4) y0: Initial condition of the full system. Needs to be provided directly or stored in model.y0
        5) z0: Initial condition of the surrogate model. Needs to be provided directly, stored in surrogateModel.z0, or
           computable via z0 = model.observable(y0)
        6) T: Final time of the MPC problem. If T is greater than the length of the reference trajectory, it is
           automatically shortened and a warning is given
        7 - 10) Q, L, R, S (type matrices (np.array) of size (dimZ, dimZ) for Q, (dimZ, 1) for L, and (dimU, dimU) for
               R, and S): Weighting of the different terms in the objective function
        11) savePath (type string): If provided, then the results are stored in the prescribed file after each iteration
        12) updateSurrogate (default: False): If true AND "createSurrogateModel(modelData, data)" is given in the
            surrogate model file, then the model is updated in each MPC loop
        13) iuInit: Number of steps to perform before starting MPC (sometimes useful for delay coordinates etc.)

        Output
        1) result (type ClassResult)

        Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

        First created: 05/2020
        """

        print('Executing ClassMPC.run with y0 = ' + str(y0) + '; z0 = ' + str(z0) + '; Q = ' +
              str(Q) + '; R = ' + str(R) + '; S = ' + str(S) + '; savePath = ' + str(savePath))

        # Obtain y0 from model if not given directly
        if y0 is None:
            y0 = model.y0

        if type(y0) == type(list()):
            print(y0, type(y0))
            y0 = np.array(y0)
            print(y0, type(y0))

        if T is None:
            T = self.T

        if reference.T < T + model.h * self.np:
            T = reference.T
            print('Warning: Reference trajectory is shorter than desired control horizon. Shortening T to ' + str(T))

        self.dimZ = model.dimZ

        self.Q = transformToMatrix(Q, model.dimZ)
        self.L = transformToVector(L, model.dimZ)
        self.R = transformToMatrix(R, model.dimU)
        self.S = transformToMatrix(S, model.dimU)

        nt = int(np.round(T / model.h))

        time = 0.0
        iTime = 0

        # Surrogate-based MPC
        if surrogateModel is not None:

            print('Solving MPC problem using surrogate model')

            surrogateModel.hShM = int(np.round(surrogateModel.h / model.h))

            result = ClassResult(self.typeOpt, nt, surrogateModel.dimU, model.nU, dimZ=surrogateModel.dimZ,
                                 flagSurrogate=True)

            if model.calcJ is not None:
                self.calcJ = model.calcJ

            if self.typeOpt == 'combinatorial':
                uTmp = np.zeros([surrogateModel.nU, 1], dtype=int)
                uTmp[:, 0] = np.array(range(surrogateModel.nU))
            else:
                if model.nU <= 2:
                    self.bounds = None
                else:
                    self.bounds = Bounds(np.zeros([self.np * (model.nU - 1)], dtype=float),
                                         np.ones([self.np * (model.nU - 1)], dtype=float))

                A = np.zeros([self.np, self.np * (model.nU - 1)], dtype=float)
                for i in range(self.np):
                    A[i, i * (model.nU - 1): (i + 1) * (model.nU - 1)] = 1.0
                self.linConstraints = LinearConstraint(A, np.zeros([self.np], dtype=float),
                                                       np.ones([self.np], dtype=float))

                test = int(iuInit) * np.ones([self.np, 1], dtype=int)
                alpha0 = mapIuToAlpha(test, surrogateModel.nU)

            # Obtain z0 either by taking it from the surrogate model or by observing y0
            if z0 is None:
                if surrogateModel.z0 is not None:
                    z0 = surrogateModel.z0
                elif (y0 is not None) and (model.observable is not None):
                    z0 = model.observable(y0)
                else:
                    print('Error in "ClassMPC.run": Initial condition z0 not specified')

            # If delay is activated, then perform model simulation until a sufficiently long history exists
            if surrogateModel.nDelay > 0:
                iuInit = iuInit * np.ones([surrogateModel.nDelay * surrogateModel.hShM + 1, 1], dtype=int)
                uInit = mapIuToU(iuInit, surrogateModel.uGrid)

                [yInit, zInit, tInit, model] = model.integrate(y0, uInit, time)

                # Update variables
                time = time + float(surrogateModel.nDelay * surrogateModel.hShM) * model.h
                iTime = iTime + surrogateModel.nDelay * surrogateModel.hShM

                t0 = np.zeros((1,))

                if not len(yInit) == 0:
                    yInit = np.concatenate((np.reshape(y0, [1, yInit.shape[1]]), yInit), axis=0)
                zInit = np.concatenate(([z0], zInit), axis=0)
                tInit = np.concatenate((t0, tInit), axis=0)

                z0 = stackZ0(zInit, surrogateModel)
                if len(yInit) > 0:
                    y0 = yInit[iTime, :]

                result.add(0, model.writeY, surrogateModel.nDelay * surrogateModel.hShM, yInit, zInit, tInit,
                           uInit, iuInit, 0.0, 1, init=True)

            else:
                if model.writeY:
                    result.add(0, model.writeY, 0, np.reshape(y0, [1, y0.shape[0]]), np.reshape(z0, [1, z0.shape[0]]),
                               np.zeros((1,)), None, None, None, None, init=True)
                else:
                    result.add(0, model.writeY, 0, None, np.reshape(z0, [1, z0.shape[0]]), np.zeros((1.0,)),
                               None, None, None, None, init=True)

            alphaOptC, omegaOptC = None, None

            # Time loop
            while iTime <= nt - self.nc * surrogateModel.hShM:

                print('t = {:.4f} sec: Solving optimization problem'.format(time))

                # Solve optimization problem
                if self.typeOpt == 'combinatorial':
                    JOpt, uOpt = 1e10, 0
                    nFev = surrogateModel.nU ** self.np
                    for iuu in itertools.product(uTmp, repeat=self.np):
                        iu = np.stack(iuu)
                        zRef = reference.z[iTime: iTime + surrogateModel.hShM * (self.nch * self.np) +
                                           surrogateModel.hShM: surrogateModel.hShM, :]
                        J = self.objectiveSurrogate(surrogateModel, zRef, mapIuToAlpha(iu, model.nU), z0, time)
                        if J < JOpt:
                            JOpt = J
                            iuOpt = iu

                    uOpt = mapIuToU(iuOpt, surrogateModel.uGrid)
                    uOptC = self.repeatControl(uOpt[:self.nc, :], self.nch * surrogateModel.hShM)
                    iuOptC = self.repeatControl(iuOpt[:self.nc, :], self.nch * surrogateModel.hShM)

                else:
                    zRef = reference.z[iTime: iTime + surrogateModel.hShM * (self.nch * self.np) +
                                       surrogateModel.hShM: surrogateModel.hShM, :]
                    JOpt, alphaOpt, nFev = self.solveOptSurrogate(surrogateModel, zRef, alpha0, z0, time)

                    alphaOptC = self.repeatControl(alphaOpt[:self.nc, :], self.nch * surrogateModel.hShM)

                    if self.typeOpt == 'SUR':

                        iuOptC, omegaOptC = sumUpRounding(alphaOptC, result.alpha[:iTime, :], result.omega[:iTime, :],
                                                          surrogateModel.nU)
                        uOptC = surrogateModel.mapAlphaToU(omegaOptC[:, :-1])

                    elif self.typeOpt == 'SUR_coarse':

                        iuOptC, omegaOptC = sumUpRoundingCoarse(alphaOptC, result.alpha[:iTime, :],
                                                                result.omega[:iTime, :], surrogateModel.nU,
                                                                self.nch * surrogateModel.hShM)
                        uOptC = surrogateModel.mapAlphaToU(omegaOptC[:, :-1])

                    else:
                        uOptC = surrogateModel.mapAlphaToU(alphaOptC)
                        iuOptC = mapUToIu(uOptC, surrogateModel.uGrid)

                    alpha0[:-1, :] = alphaOpt[1:, :]
                    alpha0[-1, :] = alphaOpt[-1, :]

                # Apply control to plant
                uOptC = np.vstack((uOptC, uOptC[-1, :]))
                iuOptC = np.vstack((iuOptC, iuOptC[-1, :]))
                [yOpt, zOpt, tOpt, model] = model.integrate(y0, uOptC, time)

                if surrogateModel.calcJ is None:
                    deltaZ = zOpt - reference.z[iTime: iTime + surrogateModel.hShM * (self.nch * self.nc) + 1, :]
                    dZ = surrogateModel.h * np.sum(np.diag(deltaZ @ self.Q @ deltaZ.T))
                    print('- Opt solved; uOpt = {}; JOpt = {}; JReal = {}; nFev = {}'.format(uOptC[0, :], JOpt, dZ, nFev))
                else:
                    print('- Opt solved; uOpt = {}; JOpt = {}; nFev = {}'.format(uOptC[0, :], JOpt, nFev))

                # Store data to output array
                result.add(iTime, model.writeY, self.nch * self.nc * surrogateModel.hShM, yOpt, zOpt, tOpt, uOptC,
                           iuOptC, JOpt, nFev, alphaOptC, omegaOptC)

                # Save data to file
                if savePath is not None:
                    result.save(savePath)
                    result.saveMat(savePath)

                # Update surrogate model
                if (surrogateModel.updateSurrogateModel is not None) and updateSurrogate:

                    zStack = result.z[iTime - surrogateModel.nDelay * surrogateModel.hShM:
                                      iTime + self.nch * self.nc * surrogateModel.hShM + 1, :]
                    uStack = result.u[iTime - surrogateModel.nDelay * surrogateModel.hShM:
                                      iTime + self.nch * self.nc * surrogateModel.hShM, :]
                    iuStack = result.iu[iTime - surrogateModel.nDelay * surrogateModel.hShM:
                                        iTime + self.nch * self.nc * surrogateModel.hShM, :]

                    surrogateModel.modelData, updatePerformed = surrogateModel.updateSurrogateModel(
                        surrogateModel.modelData, zStack, uStack, iuStack)

                    if updatePerformed:
                        print('Surrogate model has been updated')

                # Update variables
                time = time + self.nch * float(surrogateModel.hShM) * model.h
                iTime = iTime + self.nch * surrogateModel.hShM

                z0 = stackZ0(result.z[iTime - surrogateModel.hShM * surrogateModel.nDelay:iTime+1, :], surrogateModel)
                if model.writeY:
                    y0 = yOpt[-1, :]

            print('MPC Done')

        return result

    def solveOptSurrogate(self, surrogateModel, zRef, alpha0, z0, t0):

        def obj(aa):
            return self.objectiveSurrogate(surrogateModel, zRef, aa.reshape((self.np, surrogateModel.nU - 1)), z0, t0)

        # Create 1D Vector
        alpha0 = alpha0.reshape(((surrogateModel.nU - 1) * self.np, 1))

        res = minimize(obj, alpha0[:, 0], method=self.scipyMinimizeMethod, bounds=self.bounds,
                       constraints=self.linConstraints, options=self.scipyMinimizeOptions)

        return res.fun, res.x.reshape((self.np, surrogateModel.nU - 1)), res.nfev

    def objectiveSurrogate(self, surrogateModel, zRef, alpha, z0, t0):

        if self.nch > 1:
            alpha = self.repeatControl(alpha)

        # Add column of alpha in order to use all entries of the given control input
        alpha = np.concatenate((alpha, np.zeros([1, alpha.shape[1]], dtype=float)), axis=0)

        z = surrogateModel.integrateRelaxedTimeTMap(z0, t0, alpha)

        return self.calcJ(z, zRef, surrogateModel.mapAlphaToU(alpha), surrogateModel.h, surrogateModel.modelData)

    def calcJInternal(self, z, zRef, u, h, modelData=None):
        deltaZ = z[:, :self.dimZ] - zRef
        deltaU = (u[1:] - u[:-1]) / h
        return np.sum(np.diag(deltaZ @ self.Q @ deltaZ.T) + (deltaZ @ self.L).T) + np.sum(np.diag(u @ self.R @ u.T)) + \
               np.sum(np.diag(deltaU @ self.S @ deltaU.T))

    def repeatControl(self, u, nch=None):
        if nch is None:
            nch = self.nch

        uOut = np.zeros([u.shape[0] * nch, u.shape[1]])
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                uOut[i * nch: (i + 1) * nch, j] = u[i, j]
        return uOut


class ClassResult:
    y, z, u, J, t, nFev, iu, alpha, omega = [], [], [], [], [], [], [], [], []

    def __init__(self, typeOpt, nT, dimU, nU, dimY=None, dimZ=None, flagSurrogate=False):
        if dimY is not None:
            self.y = np.zeros([nT + 1, dimY], dtype=float)

        self.z = np.zeros([nT + 1, dimZ], dtype=float)
        self.u = np.zeros([nT + 1, dimU], dtype=float)
        self.iu = np.zeros([nT + 1, 1], dtype=float)
        self.J = np.zeros([nT + 1, 1], dtype=float)
        self.t = np.zeros([nT + 1, 1], dtype=float)
        self.nFev = np.zeros([nT + 1, 1], dtype=float)

        if (typeOpt != 'combinatorial') and (flagSurrogate is True):
            self.alpha = np.zeros([nT + 1, nU - 1], dtype=float)

        if (typeOpt == 'SUR') or (typeOpt == 'SUR_coarse'):
            self.omega = np.zeros([nT + 1, nU], dtype=float)

    def add(self, iTime, writeY, nWrite, y, z, t, u, iu, J, nFev, alpha=None, omega=None, init=False):

        iEnd = min(iTime + nWrite + 1, self.t.shape[0])
        nWrite -= (iTime + nWrite) - iEnd

        if init:
            if writeY:
                dimY = y.shape[1]
                self.y = np.zeros([self.t.shape[0], dimY], dtype=float)
                self.y[0, :] = y[0, :]

            dimZ = z.shape[1]
            self.z = np.zeros([self.t.shape[0], dimZ], dtype=float)
            self.z[0, :] = z[0, :]

        if nWrite > 1:
            if writeY:
                self.y[iTime: iEnd, :] = y[:nWrite, :]

            self.z[iTime: iEnd, :] = z[:nWrite, :]
            self.t[iTime: iEnd, 0] = t[:nWrite]

            self.u[iTime: iEnd, :] = u[:nWrite, :]
            self.iu[iTime: iEnd, 0] = iu[:nWrite, 0]
            self.J[iTime: iEnd, :] = J
            self.nFev[iTime: iEnd, :] = nFev
            if alpha is not None:
                self.alpha[iTime: iEnd - 1, :] = alpha[:nWrite - 1, :]
            if omega is not None:
                self.omega[iTime: iEnd - 1, :] = omega[:nWrite - 1, :]

    def save(self, savePath):
        np.savez(savePath, y=self.y, z=self.z, u=self.u, J=self.J, t=self.t, nFev=self.nFev, iu=self.iu,
                 alpha=self.alpha, omega=self.omega)

    def saveMat(self, savePath):
        savemat(savePath, {'y': self.y, 'z': self.z, 'u': self.u, 'J': self.J, 't': self.t, 'nFev': self.nFev,
                           'iu': self.iu, 'alpha': self.alpha, 'omega': self.omega},
                appendmat=True)

    def load(self, loadPath):
        if loadPath[-4:] != '.npz':
            loadPath = loadPath + '.npz'
        dataIn = np.load(loadPath)

        self.y = dataIn['y']
        self.z = dataIn['z']
        self.u = dataIn['u']
        self.J = dataIn['J']
        self.t = dataIn['t']
        self.nFev = dataIn['nFev']
        self.iu = dataIn['iu']
        self.alpha = dataIn['alpha']
        self.omega = dataIn['omega']


def sumUpRounding(alphaNew, alpha, omega, nU):
    # in some SUR papers, alpha = q and omega = p

    omegaOut = np.zeros([alphaNew.shape[0], nU], dtype=float)
    iOut = np.zeros([alphaNew.shape[0], 1], dtype=int)

    omegaHat = np.zeros([nU], dtype=float)
    for j in range(alphaNew.shape[0]):
        for i in range(nU - 1):
            omegaHat[i] = np.sum(alphaNew[:j + 1, i]) + np.sum(alpha[:, i]) - (
                    np.sum(omegaOut[:j, i]) + np.sum(omega[:, i]))

        omegaHat[nU - 1] = np.sum(1.0 - np.sum(alphaNew[:j + 1, :], axis=1)) + np.sum(
            1.0 - np.sum(alpha, axis=1)) - (np.sum(omegaOut[:j, -1]) + np.sum(omega[:, -1]))

        iOut[j, 0] = np.argmax(omegaHat)
        omegaOut[j, iOut[j, 0]] = 1.0

    return iOut, omegaOut


def sumUpRoundingCoarse(alphaNew, alpha, omega, nU, nConst):
    # in some SUR papers, alpha = q and omega = p

    omegaOut = np.zeros([alphaNew.shape[0], nU], dtype=float)
    iOut = np.zeros([alphaNew.shape[0], 1], dtype=int)

    omegaHat = np.zeros([1, nU], dtype=float)
    indices = range(0, alphaNew.shape[0], nConst)
    for j in indices:
        for i in range(nU - 1):
            omegaHat[0, i] = np.sum(alphaNew[:j:nConst + 1, i]) + np.sum(alpha[::nConst, i]) - (
                    np.sum(omegaOut[:j:nConst, i]) + np.sum(omega[::nConst, i]))

        omegaHat[0, nU - 1] = np.sum(1.0 - np.sum(alphaNew[:j + 1: nConst, :], axis=1)) + np.sum(
            1.0 - np.sum(alpha[::nConst, :], axis=1)) - (np.sum(omegaOut[:j:nConst, -1]) + np.sum(omega[::nConst, -1]))

        if j == alphaNew.shape[0]:
            iOut[j, 0] = np.argmax(omegaHat)
            omegaOut[j, iOut[j, 0]] = 1.0
        else:
            iOut[j: j + nConst, 0] = np.argmax(omegaHat)
            omegaOut[j: j + nConst, iOut[j, 0]] = 1.0

    return iOut, omegaOut


def mapIuToU(iu, uGrid):
    if len(iu.shape) > 1:
        u = np.zeros([iu.shape[0], uGrid.shape[1]], dtype=float)
        for i in range(iu.shape[0]):
            u[i, :] = uGrid[iu[i, 0], :]
    elif len(iu.shape) == 1:
        u = np.zeros([iu.shape[0], uGrid.shape[1]], dtype=float)
        for i in range(iu.shape[0]):
            u[i, :] = uGrid[iu[i], :]
    else:
        u = np.zeros([1, uGrid.shape[1]], dtype=float)
        u[0, :] = uGrid[iu, :]

    return u


def mapIuToAlpha(iu, nU):
    alpha = np.zeros([iu.shape[0], nU], dtype=float)
    for i in range(iu.shape[0]):
        alpha[i, iu[i, 0]] = 1.0

    return alpha[:, :-1]


def mapUToIu(u, uGrid):
    iu = -np.ones([u.shape[0], 1], dtype=int)
    for i in range(u.shape[0]):
        for j in range(uGrid.shape[0]):
            if np.linalg.norm(u[i, :] - uGrid[j, :]) < 1e-4:
                iu[i, 0] = j
                break

    return iu


def transformToMatrix(A, dim):
    if type(A) == type(sparse.spdiags(1.0, [0], 1, 1)):
        return A

    A = np.asarray(A)
    if A.size == 1:
        B = np.identity(dim, dtype=float)
        for i in range(dim):
            B[i, i] = A
    else:
        if A.ndim == 1:
            if A.shape[0] == dim:
                B = np.diagflat(A)
            else:
                print('Error in "ClassReferenceTrajectory.transformToMatrix": please specify Q, R and S as '
                      'scalars, as arrays of length dimZ (for Q) or dimU (for R and S) containing the diagonal '
                      'entries of Q, R, and S, respectively, or as square matrices with dimension (dimZ x dimZ)'
                      '(for Q) or (dimU x dimU) (for R and S) ')
                exit(1)
        else:
            if (A.shape[0] == A.shape[1]) and (A.shape[0] == dim):
                B = A
            elif (A.shape[0] == dim) and (A.shape[1] == 1):
                B = np.diagflat(A)
            elif (A.shape[0] == 1) and (A.shape[1] == dim):
                B = np.diagflat(A)
            else:
                print('Error in "ClassReferenceTrajectory.transformToMatrix": please specify Q, R and S as '
                      'scalars, as arrays of length dimZ (for Q) or dimU (for R and S) containing the diagonal '
                      'entries of Q, R, and S, respectively, or as square matrices with dimension (dimZ x dimZ)'
                      '(for Q) or (dimU x dimU) (for R and S) ')
                exit(1)

    return B


def transformToVector(A, dim):
    A = np.asarray(A)
    if A.size == 1:
        B = A * np.ones([dim, 1], dtype=float)
    else:
        if A.ndim == 1:
            if A.shape[0] == dim:
                B = np.zeros([dim, 1], dtype=float)
                B[:, 0] = A
            else:
                print('Error in "ClassReferenceTrajectory.transformToVector": please specify L as a '
                      'scalar or array of length dimZ ')
                exit(1)
        else:
            if (A.shape[0] == dim) and (A.shape[1] == 1):
                B = A
            elif (A.shape[0] == 1) and (A.shape[1] == 0):
                B = A.T
            else:
                print('Error in "ClassReferenceTrajectory.transformToVector": please specify L as a '
                      'scalar or array of length dimZ ')
                exit(1)

    return B


def stackZ0(z, surrogateModel):
    z0 = np.zeros([1, surrogateModel.dimZ * (1 + surrogateModel.nDelay)], dtype=float)
    for i in range(surrogateModel.nDelay + 1):
        z0[0, i * surrogateModel.dimZ: (i + 1) * surrogateModel.dimZ] = z[-(i * surrogateModel.hShM + 1), :]
    return z0
