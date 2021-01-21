from sys import path
from os import getcwd, sep
path.append(getcwd()[:getcwd().rfind(sep)])


import OpenFOAM

from OpenFOAM.classesOpenFOAM import *
from visualization import *

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define model
# -------------------------------------------------------------------------------------------------------------------- #
pathProblem = 'OpenFOAM/problems/cylinder'
pathOut = 'tests/results/cylinder_LSTM'

pathData_ref = pathOut + '/new/data_5_grid2'
pathData = pathOut + '/new/data_5_grid2'
pathSurrogate = pathOut + '/new/Surrogate/surrogate_5_grid2'

reuseSurrogate = True
nProc = 8

nInputs = 1
dimInputs = 1
iInputs = [0]

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define what to observe
# -------------------------------------------------------------------------------------------------------------------- #

writeGrad = False
writeY = False

forceCoeffsPatches = ['cylinder']
ARef = 1.0
lRef = 1.0

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
Re = 100.0
hFOAM = 0.01
dimSpace = 2

T = 20.1
h = 0.05

uMin = [-5.0]
uMax = [5.0]
nGridU = 2  # number of parts the grid is split into
# (uMin = [-2.0], uMax = [2.0], nGridU = 2 --> uGrid = [-2, 0, 2])

dimZ = 2

Ttrain = 500.0
nLag = 2  # Lag time for LSTM
nDelay = 15  # Number of delays for modeling
nhidden = 500 # number of hidden neurons in LSTM cell
epochs = 2 # Trainin epochs
batch_size = 75 # batch size for LSTM - training


# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

obs = ClassObservable(forceCoeffsPatches=forceCoeffsPatches, ARef=ARef, lRef=lRef, writeY=writeY, writeGrad=writeGrad)

of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, dimZ=dimZ, typeUGrid='cube', nGridU=nGridU)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls

uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=nLag*2, nhMax=nLag*5)
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=nLag*10, nhMax=nLag*10,u=uTrain,iu=iuTrain)

# Create a data set (and save it to an npz file)
if os.path.exists(pathData + '.npz'):
    print("reuse data")
    dataSet.createData(loadPath=pathData)
else:
    dataSet.createData(model=model, u=uTrain, savePath=pathData)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y', rawData=dataSet.rawData, nLag=nLag, nDelay=nDelay)

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

z0 = dataSet.rawData.z[0][0, :]
surrogate = ClassSurrogateModel('LSTM.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ, z0=z0, nDelay=nDelay,
                                nhidden=nhidden, epochs=epochs, nLag=nLag,batch_size=batch_size)

if os.path.exists(pathSurrogate + '.pkl') and reuseSurrogate:
    surrogate.createROM(data, loadPath=pathSurrogate)
else:
    surrogate.createROM(data, savePath=pathSurrogate)

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory
TRef = T + 20.0
nRef = int(round(TRef / h)) + 1
t = np.linspace(0, T, nRef)
sinRef = np.sin(t)
zRef = np.zeros([nRef, 2], dtype=float)
zRef[:,1] = sinRef

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef)

# Create class for the MPC problem
options = {'eps': 1e-9}

MPC = ClassMPC(np=5, nc=1, nch=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP', scipyMinimizeOptions=options)

# Weights for the objective function
Q = [0.0, 1.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
L = [0.0, 0.0]  # reference tracking: (z - deltaZ)^T * L -> want to minimize mean lift
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
save_path_cont = pathOut + 'LSTM_5_grid2_lift'
save_path_SUR = pathOut + 'LSTM_5_grid2_lift_SUR'


resultCont = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S, L=L, updateSurrogate=False, savePath=save_path_cont,iuInit=1)

plot(z={'t': resultCont.t[1:], 'z': resultCont.z[1:], 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t[1:], 'u': resultCont.u[1:], 'iplot': 1},
     J={'t': resultCont.t[1:], 'J': resultCont.J[1:], 'iplot': 2},
     nFev={'t': resultCont.t[1:], 'nFev': resultCont.nFev[1:], 'iplot': 3},
     fOut=pathOut + '/MPC1')

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR_coarse'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S,savePath=save_path_SUR,iuInit=1)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5},
     fOut=pathOut + '/MPC2')

print('Done')
