from QuaSiModO import *
from visualization import *


# -------------------------------------------------------------------------------------------------------------------- #
# ODE: Define right-hand side as a a function of the state y and the control u
# -------------------------------------------------------------------------------------------------------------------- #
def rhs(y_, u_):
    nu, lam, delta = -0.05, -1.0, 1.0
    return np.array([nu * y_[0], lam * (y_[1] - y_[0] * y_[0]) + delta * 2.0 * pow(u_[0], 1.0)])


# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
T = 10.0  # Time for the MPC problem
h = 0.1  # Time step for the ODE solver, for the training data sampling and for the MPC step length
y0 = [1.0, 2.0]  # Initial condition of the ODE
uMin = [-2.0]  # Minimal value of the input (also defines dimU)
uMax = [2.0]  # Maximal value of the input (also defines dimU)

dimZ = 2  # dimension of the observable (= dimY in the ODE case, unless iObs is passed to ClassModel)
nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])

Ttrain = 100.0  # Time for the simulation in the traing data generation
nLag = 5  # Lag time for EDMD
nMonomials = 2  # Max order of monomials for EDMD

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
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=10, nhMax=20)
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=[2.0], u=uTrain, iu=iuTrain)

# Create a data set (and save it to an npz file)
dataSet.createData(model=model, y0=y0, u=uTrain)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y', rawData=dataSet.rawData, nLag=nLag, nDelay=0)

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ, z0=y0, nDelay=0,
                                nLag=nLag, nMonomials=nMonomials, epsUpdate=0.05)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 1], dtype=float)

# zRef[:, 0] = 3.0
tRef = np.array(np.linspace(0.0, T, nRef))
zRef[:, 0] = 1.0 + np.sin(2.0 * tRef * 2.0 * np.pi / TRef)

iRef = [1]

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)

# Create class for the MPC problem
MPC = ClassMPC(np=3, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  # scipyMinimizeMethod=

# Weights for the objective function
Q = [0.0, 1.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t[:-1], 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t[:-1], 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t[:-1], 'nFev': resultCont.nFev, 'iplot': 3})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t[:-1], 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t[:-1], 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t[:-1], 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t[:-1], 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t[:-1], 'omega': result_SUR.omega, 'iplot': 5})
