# -------------------------------------------------------------------------------------------------------------------- #
# Add path and create output folder
from os import sep, makedirs, path
from sys import path as syspath

# Add path
fileName = path.abspath(__file__)
pathMain = fileName[:fileName.find(sep + 'QuaSiModO') + 10]
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
pathProblem = path.join(pathMain, 'OpenFOAM/problems/cylinder')
pathData = path.join(pathOut, 'data')
nProc = 1

nInputs = 1
dimInputs = 1
iInputs = [0]

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define what to observe
# -------------------------------------------------------------------------------------------------------------------- #

probeLocations = [[7.0, 7.5, 0.5], [8.0, 7.5, 0.5], [9.0, 7.5, 0.5], [10.0, 7.5, 0.5],
                  [7.0, 8.5, 0.5], [8.0, 8.5, 0.5], [9.0, 8.5, 0.5], [10.0, 8.5, 0.5],
                  [7.0, 6.5, 0.5], [8.0, 6.5, 0.5], [9.0, 6.5, 0.5], [10.0, 6.5, 0.5]]
probeQuantities = ['U']
probeDimensions = [3]
probeQuantitiesIndices = [[0, 1]]

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
Re = 100.0
hFOAM = 0.01
dimSpace = 2

T = 60.0
h = 0.05

uMin = [-2.0]
uMax = [2.0]
nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])

Ttrain = 100.0  # Time for the simulation in the traing data generation
nLag = 1  # Lag time for EDMD
nDelay = 2
nMonomials = 0  # Max order of monomials for EDMD

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

obs = ClassObservable(probeLocations=probeLocations, probeQuantities=probeQuantities, probeDimensions=probeDimensions,
                      probeQuantitiesIndices=probeQuantitiesIndices)

of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, typeUGrid='cube', nGridU=nGridU)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=2, nhMax=10)

# Create a data set (and save it to an npz file)
if os.path.exists(pathData + '.npz'):
    dataSet.createData(loadPath=pathData)
else:
    dataSet.createData(model=model, u=uTrain, savePath=pathData)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y', rawData=dataSet.rawData, nDelay=nDelay)

model.dimZ = dataSet.dimZ

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

z0 = dataSet.rawData.z[0][0, :]
surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, h=model.h, dimZ=z0.shape[0], z0=z0, nDelay=nDelay,
                                nLag=nLag, nMonomials=nMonomials, of=of, Re=Re, epsUpdate=0.05)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, int(model.dimZ / 2)], dtype=float)
iRef = np.array(range(1, model.dimZ, 2))
reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)

# Create class for the MPC problem
MPC = ClassMPC(np=20, nc=1, nch=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP') # scipyMinimizeMethod='trust-constr'

# Weights for the objective function
Q = np.zeros([model.dimZ, 1], dtype=float)
Q[1::2, 0] = 1.0
R = [0.0]  # control cost: u^T * R * u
S = [10.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S, updateSurrogate=True)
resultCont.saveMat('MPC-Cont', pathOut)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S, updateSurrogate=True)
result_SUR.saveMat('MPC-SUR', pathOut)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})

print('Done')
