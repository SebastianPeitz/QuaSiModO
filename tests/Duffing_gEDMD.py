# -------------------------------------------------------------------------------------------------------------------- #
# Add path and create output folder
from os import sep, makedirs, path
from sys import path as syspath

# Add path
fileName = path.abspath(__file__)
pathMain = fileName[:fileName.lower().find(sep + 'quasimodo') + 10]
syspath.append(pathMain)

# Create output folder
pathOut = path.join(pathMain, 'tests', 'results', fileName[fileName.rfind(sep) + 1:-3])
makedirs(pathOut, exist_ok=True)
# -------------------------------------------------------------------------------------------------------------------- #

from QuaSiModO import *
from visualization import *


# -------------------------------------------------------------------------------------------------------------------- #
# ODE: Define right-hand side as a a function of the state y and the control u
# -------------------------------------------------------------------------------------------------------------------- #
def rhs(y_, u_):
    alpha, beta, delta = -1.0, 1.0, 0.0
    return np.array([y_[1], -delta * y_[1] - alpha * y_[0] - beta * y_[0] * y_[0] * y_[0] + u_[0]])


# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
T = 50.0  # Time for the MPC problem
h = 0.005  # Time step for the ODE solver, for the training data sampling and for the MPC step length
y0 = [1.0, 1.0]  # Initial condition of the ODE
uMin = [-2.0]  # Minimal value of the input (also defines dimU)
uMax = [2.0]  # Maximal value of the input (also defines dimU)

dimZ = 2  # dimension of the observable (= dimY in the ODE case, unless iObs is passed to ClassModel)
nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])

Ttrain = 60.0  # Time for the simulation in the traing data generation
nLag = 20  # Lag time for EDMD
nMonomials = 5  # Max order of monomials for EDMD

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

# Create model class
model = ClassModel(rhs, h=h, uMin=uMin, uMax=uMax, dimZ=dimZ, typeUGrid='cube', nGridU=nGridU)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=1, nhMax=5)

# Create a data set (and save it to an npz file)
y0Train = list()
for i in range(5):
    y0Train.append(np.array([-2.0, -2.0]) + 2.0 * np.random.rand((2)))
dataSet.createData(model=model, y0=y0Train, u=uTrain)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='dX', rawData=dataSet.rawData, nLag=nLag, nDelay=0)

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

surrogate = ClassSurrogateModel('gEDMD.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ, z0=y0, nDelay=0,
                                nLag=nLag, nMonomials=nMonomials, epsUpdate=0.05)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 2], dtype=float)

# zRef[:, 0] = 3.0
tRef = np.array(np.linspace(0.0, T, nRef))
zRef[np.where(tRef >= 10.0), 0] = 1.0
zRef[np.where(tRef >= 20.0), 0] = -1.0
zRef[np.where(tRef >= 30.0), 0] = 0.5
zRef[np.where(tRef >= 40.0), 0] = -1.0

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef)

# Create class for the MPC problem
MPC = ClassMPC(np=3, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  # scipyMinimizeMethod=

# Weights for the objective function
Q = [5.0, 1.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)
resultCont.saveMat('MPC-Cont', pathOut)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR_coarse'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)
result_SUR.saveMat('MPC-SUR', pathOut)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})
