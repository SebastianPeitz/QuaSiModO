from sys import path
from os import getcwd, sep
path.append(getcwd()[:getcwd().rfind(sep)])


from QuaSiModO import *
from visualization import *
import os


# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
T = 20.0  # Time for the MPC problem
h = 0.05  # Time step for the ODE solver, for the training data sampling and for the MPC step length

uMin = [-0.2]  # Minimal value of the input (also defines dimU)
uMax = [2.0]  # Maximal value of the input (also defines dimU)

nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])
uGrid = np.array([uMin, [0.0], uMax])

Ttrain = 2.0  # Time for the simulation in the traing data generation
nLag = 5  # Lag time for EDMD
nMonomials = 2  # Max order of monomials for EDMD

tau = 2.0
dimZ = 5

y0 = list()
nSpline = 4
tSpline = np.linspace(-tau, 0.0, nSpline)
tTau = np.linspace(-tau, 0.0, int(tau/h) + 1)
for i in range(100):
    tck = interpolate.splrep(tSpline, 0.5 + 2.0 * np.random.rand(nSpline), s=0)
    y0.append(interpolate.splev(tTau, tck, der=0))
    # y0.append(np.linspace(0.5 + 2.0 * np.random.rand(1), 0.5 + 2.0 * np.random.rand(1), int(tau/h) + 1)[:, 0])
# y0 = np.linspace(0.5, 1.0, int(tau/h) + 1)
params = {'tau': tau}

pathData = 'tests/results/MackeyGlass/data'

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

# Create model class
model = ClassModel('mackey-glass.py', h=h, uMin=uMin, uMax=uMax, dimZ=dimZ, params=params, typeUGrid='cube', nGridU=nGridU, uGrid=uGrid)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=5, nhMax=20)
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=5, nhMax=20, u=uTrain, iu=iuTrain)
# uTrain, iuTrain = None, None
# for i in range(model.nU):
#     uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=model.uGrid[i], u=uTrain, iu=iuTrain)

# Create a data set (and save it to an npz file)
if not os.path.exists(pathData):
     os.makedirs(pathData)

if os.path.exists(pathData + '.npz'):
    dataSet.createData(loadPath=pathData)
else:
     dataSet.createData(model=model, y0=y0, u=uTrain, savePath=pathData)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y', rawData=dataSet.rawData, nLag=nLag, nDelay=0)

#plotPhase3D(dataSet.rawData.z[0][:, 0], dataSet.rawData.z[0][:, 2], dataSet.rawData.z[0][:, 4])

y0 = dataSet.rawData.y[0][-1, :]
z0 = dataSet.rawData.z[0][-1, :]

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ,
                                z0=z0, nDelay=0, nLag=nLag, nMonomials=nMonomials, epsUpdate=0.05)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

# Simulate surrogate model using every nLag'th entry of the training input
Tcomp = 5.0
tS = np.linspace(0.0, Tcomp, int(Tcomp / (nLag * model.h)) + 1)
tF = np.linspace(0.0, Tcomp, int(Tcomp / (model.h)) + 1)
zS = list()
zF = list()
for i in range(model.nU):

    iu = i * np.ones([len(tS)-1, 1], dtype=int)
    [z, tSurrogate] = surrogate.integrateDiscreteInput(z0, 0.0, iu)
    zS.append(z)

    u = model.uGrid[i, 0] * np.ones([len(tF)-1, 1], dtype=int)
    [yOpt, zOpt, tOpt, model] = model.integrate(y0, u, 0.0)
    zF.append(zOpt)

# Compare states and control
plot(z0={'t': tOpt, 'z0': zF[0], 'iplot': 0},
     z0r={'t': tSurrogate, 'z0r': zS[0], 'iplot': 0},
     z1={'t': tOpt, 'z1': zF[1], 'iplot': 1},
     z1r={'t': tSurrogate, 'z1r': zS[1], 'iplot': 1},
     z2={'t': tOpt, 'z2': zF[2], 'iplot': 2},
     z2r={'t': tSurrogate, 'z2r': zS[2], 'iplot': 2}
     )

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 5.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 1], dtype=float)

# zRef[:, 0] = 3.0
tRef = np.array(np.linspace(0.0, T, nRef))
zRef[:, 0] = 1.0  # + np.sin(2.0 * tRef * 2.0 * np.pi / TRef)

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef)

# Create class for the MPC problem
MPC = ClassMPC(np=10, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  # scipyMinimizeMethod=

# Weights for the objective function
Q = [1.0, 0.0, 0.0, 0.0, 0.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, z0=z0, T=T, Q=Q, R=R, S=S)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, z0=z0, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})
