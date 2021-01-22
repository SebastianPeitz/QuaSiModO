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

from OpenFOAM.classesOpenFOAM import *
from visualization import *

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define model
# -------------------------------------------------------------------------------------------------------------------- #
pathProblem = path.join(pathMain, 'OpenFOAM/problems/fluidicPinball')
pathData = path.join(pathOut, 'data')
nProc = 4

nInputs = 3
dimInputs = 1
iInputs = [0]

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define what to observe
# -------------------------------------------------------------------------------------------------------------------- #

writeGrad = False
writeY = False

forceCoeffsPatches = ['Cyl1', 'Cyl2', 'Cyl3']
ARef = 1.0
lRef = 1.0

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
Re = 200.0
hFOAM = 0.005
dimSpace = 2

T = 60.0
h = 0.05

uMin = [-2.0, -2.0, -2.0]
uMax = [2.0, 2.0, 2.0]
nGridU = 1  # number of parts the grid is split into

dimZ = 6

Ttrain = 100.0  # Time for the simulation in the traing data generation
nLag = 5  # Lag time for EDMD
nDelay = 1  # Number of delays for modeling
nMonomials = 1  # Max order of monomials for EDMD

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

obs = ClassObservable(forceCoeffsPatches=forceCoeffsPatches, ARef=ARef, lRef=lRef, writeY=writeY, writeGrad=writeGrad)

of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, dimZ=dimZ, typeUGrid='cubeCenter', nGridU=nGridU)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=10, nhMax=20)
# uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=[0.0, 0.0, 0.0], u=uTrain, iu=iuTrain)

# Create a data set (and save it to an npz file)
if os.path.exists(pathData + '.npz'):
    dataSet.createData(loadPath=pathData)
else:
    dataSet.createData(model=model, u=uTrain, savePath=pathData)

# plot data
t = np.linspace(0.0, Ttrain, uTrain[0].shape[0])
plot(u0={'t': dataSet.rawData.t[0], 'u0': dataSet.rawData.u[0], 'iplot': 0},
     z0={'t': dataSet.rawData.t[0], 'z0': dataSet.rawData.z[0], 'iplot': 1}
     )

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y', rawData=dataSet.rawData, nLag=nLag, nDelay=nDelay)

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

z0 = dataSet.rawData.z[0][0, :]
surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ, z0=z0, nDelay=nDelay,
                                nLag=nLag, nMonomials=nMonomials, epsUpdate=0.05)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

# Simulate surrogate model using every nLag'th entry of the training input
iuCoarse = dataSet.rawData.iu[1][nDelay * nLag::nLag]
z0 = np.zeros([1, model.dimZ * (nDelay + 1)], dtype=float)
for i in range(nDelay + 1):
    z0[0, i * model.dimZ: (i + 1) * model.dimZ] = dataSet.rawData.z[1][(nDelay - i) * nLag, :]
[z, tSurrogate] = surrogate.integrateDiscreteInput(z0, nLag * nDelay * h, iuCoarse)

# Compare states and control
plot(z={'t': dataSet.rawData.t[1], 'z': dataSet.rawData.z[1][:, :model.dimZ], 'iplot': 0},
     zr={'t': tSurrogate, 'zr': z[:, :model.dimZ], 'markerSize': 5, 'iplot': 0})

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 3], dtype=float)

# zRef[:, 0] = 3.0
tRef = np.array(np.linspace(0.0, T, nRef))
# zRef[:, 0] = 0.0 + 1.5 * np.sin(2.0 * tRef * 2.0 * np.pi / TRef)

zRef[:, 0] = 0.5
zRef[:, 1] = -0.5

iRef = np.where(tRef > 10.0)
zRef[iRef, 0] = 1.0
zRef[iRef, 1] = -1.0

iRef = np.where(tRef > 20.0)
zRef[iRef, :] = 0.0

iRef = np.where(tRef > 30.0)
zRef[iRef, 0] = -1.0
zRef[iRef, 1] = -0.5

iRef = np.where(tRef > 40.0)
zRef[iRef, 0] = -0.5
zRef[iRef, 1] = -1.0

iRef = np.where(tRef > 50.0)
zRef[iRef, :] = 0.0

iRef = [1, 3, 5]

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)

# Create class for the MPC problem
MPC = ClassMPC(np=3, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  # scipyMinimizeMethod='trust-constr'

# Weights for the objective function
Q = [0.0, 1.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [1e-3]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
# resultCont = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S)
# resultCont.saveMat('MPC-Cont', pathOut)
#
# plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
#      u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
#      J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
#      nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR_coarse'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S)
result_SUR.saveMat('MPC-SUR', pathOut)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})

print('Done')
