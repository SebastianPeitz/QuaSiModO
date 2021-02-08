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
pathProblem = path.join(pathMain, 'OpenFOAM/problems/cylinder')
pathData = path.join(pathOut, 'data')
pathSurrogate = path.join(pathOut, 'surrogate_3')
reuseSurrogate = True
nProc = 1

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

Ttrain = 500.0  # Time for the simulation in the traing data generation
nLag = 2  # Lag time for LSTM
nDelay = 15  # Number of delays for modeling
nhidden = 500  # number of hidden neurons in LSTM cell
epochs = 2  # Trainin epochs
batch_size = 75  # batch size for LSTM - training


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
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

steps = 300
steps_sur = int(steps / nLag)
# # Simulate surrogate model using a random control
# steps_delay = nDelay * nLag
# start = 0
#
# iu = np.random.randint(0, 3, [steps_sur, 1])
# iu = np.reshape(np.concatenate((50 * [0], 50 * [1], 50 * [2])), [150, 1])
# u = np.zeros([steps, 1])
# for i in range(steps_sur):
#     u[nLag * i:nLag * (i + 1), 0] = model.uGrid[iu[i, 0]]
#
# z0 = dataSet.rawData.z[0][start, :]
#
# # compute delay
# _, z_delay, t_delay, _ = model.integrate(z0, np.zeros([steps_delay + 1, 1]), 0.0)
#
# _, z, t, _ = model.integrate(z0, u, steps_delay * model.h)
#
# temp = np.reshape(z_delay[::nLag, :], [1, (nDelay + 1), dimZ])
# temp = temp[:, ::-1, :]
# z0 = np.reshape(temp, [dimZ * (nDelay + 1)])
#
# [zSurrogate, tSurrogate] = surrogate.integrateDiscreteInput(z0, steps_delay * model.h, iu)
# print(z[0, :dimZ], t[0])
# print(zSurrogate[0, :dimZ], tSurrogate[0])
#
# fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True, figsize=(10, 6))
# plt.title("Test performance", fontsize=12)
# for i in range(2):
#     axs[i].plot(tSurrogate, zSurrogate[:, i], linewidth=2)
#     axs[i].plot(t, z[:, i], linewidth=2)
#     axs[i].set_ylabel(r"$x[$" + str(i) + r"$]$", fontsize=12)
#
# axs[2].plot(tSurrogate, model.uGrid[iu[:, 0]], linewidth=2)
# axs[2].plot(t, u, linewidth=2)
# plt.xlabel(r"$t$", fontsize=12)
#
# # Compare states and control
# plot(z={'t': t, 'z': z[:, :model.dimZ], 'iplot': 0},
#      zr={'t': tSurrogate, 'zr': zSurrogate[:, :model.dimZ], 'markerSize': 5, 'iplot': 0})

# Test 2
# Simulate surrogate model using every nLag'th entry of the training input
iuCoarse = dataSet.rawData.iu[0][nDelay * nLag::nLag]
z0 = np.zeros([1, model.dimZ * (nDelay + 1)], dtype=float)
for i in range(nDelay + 1):
    z0[0, i * model.dimZ: (i + 1) * model.dimZ] = dataSet.rawData.z[0][(nDelay - i) * nLag, :]
[z, tSurrogate] = surrogate.integrateDiscreteInput(z0, nLag * nDelay * h, iuCoarse)

# Compare states and control
plot(z={'t': dataSet.rawData.t[0][:steps], 'z': dataSet.rawData.z[0][:steps, :model.dimZ], 'iplot': 0},
     zr={'t': tSurrogate[:steps_sur], 'zr': z[:steps_sur, :model.dimZ], 'markerSize': 5, 'iplot': 0})

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

resultCont = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S, iuInit=1)
resultCont.saveMat('MPC-Cont', pathOut)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR_coarse'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S, iuInit=1)
result_SUR.saveMat('MPC-SUR', pathOut)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})

print('Done')
