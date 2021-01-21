from sys import path
from os import getcwd, sep
path.append(getcwd()[:getcwd().rfind(sep)])


from OpenFOAM.classesOpenFOAM import *
from visualization import *

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define model
# -------------------------------------------------------------------------------------------------------------------- #
pathProblem = 'OpenFOAM/problems/cylinder'
pathOut = 'tests/results/cylinderPOD'
pathData = pathOut + '/data'
pathROM = pathOut + '/ROM'
nProc = 1
BCWrite = 'BCWrite'

nInputs = 1
dimInputs = 1
iInputs = [0]

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define what to observe
# -------------------------------------------------------------------------------------------------------------------- #

nModes = 25

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
Re = 100.0
hFOAM = 0.01
dimSpace = 2

T = 60.0
h = 0.1

uMin = [-4.0]
uMax = [4.0]
nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])

Ttrain = 180.0  # Time for the simulation in the traing data generation

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

obs = ClassObservable(flagFullState=True)

of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace, BCWrite=BCWrite)
model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, typeUGrid='cube', nGridU=nGridU)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=2, nhMax=10)
# uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=[0.0], u=uTrain, iu=iuTrain)

# Create a data set (and save it to an npz file)
if os.path.exists(pathData + '.npz'):
    dataSet.createData(loadPath=pathData)
else:
    dataSet.createData(model=model, u=uTrain, savePath=pathData)

# plot data
# t = np.linspace(0.0, Ttrain, uTrain[0].shape[0])
# plot(u0={'t': dataSet.rawData.t[0], 'u0': dataSet.rawData.u[0], 'iplot': 0})

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='', rawData=dataSet.rawData)

model.dimZ = dataSet.dimZ

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

yBase, _ = of.readSolution(999.0, 999.0, 1.0, whichField='U')
yC, _ = of.readSolution(899.0, 899.0, 1.0, whichField='U')

z0 = dataSet.rawData.z[0][0, :]
surrogate = ClassSurrogateModel('POD.py', uGrid=model.uGrid, h=model.h, dimZ=z0.shape[0], z0=z0,
                                yBase=yBase.T, yC=yC.T, nModes=nModes, of=of, Re=Re)
if os.path.exists(pathROM + '.pkl'):
    surrogate.createROM(data, loadPath=pathROM)
else:
    surrogate.createROM(data, savePath=pathROM)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

pathDataValidation = pathOut + '/dataValidation'

# Create comparison data
dataSetValidation = ClassControlDataSet(h=model.h, T=30.0)
uValidate, iuValidate = None, None
for i in range(surrogate.uGrid.shape[0]):
    uValidate, iuValidate = dataSetValidation.createControlSequence(model, typeSequence=surrogate.uGrid[i, :],
                                                                    u=uValidate, iu=iuValidate)

# Create a data set (and save it to an npz file)
if os.path.exists(pathDataValidation + '.npz'):
    dataSetValidation.createData(loadPath=pathDataValidation)
else:
    dataSetValidation.createData(model=model, u=uValidate, savePath=pathDataValidation)

# data = dataSetValidation.prepareData(model, method='', rawData=dataSetValidation.rawData)
# if os.path.exists(pathROM):
#     surrogate.createROM(data, loadPath=pathROM)
# else:
#     surrogate.createROM(data, savePath=pathROM)

# Simulate surrogate model using every nLag'th entry of the training input
zPod = list()
alphaPod = list()
alphaFull = list()

for i in range(surrogate.uGrid.shape[0]):
    alphaPod.append(np.zeros([dataSetValidation.rawData.z[i].shape[0], surrogate.modelData.A[i].shape[0]], dtype=float))
    alphaFull.append(np.zeros([dataSetValidation.rawData.z[i].shape[0], surrogate.modelData.A[i].shape[0]], dtype=float))

for i in range(surrogate.uGrid.shape[0]):
    [zi, ti] = surrogate.integrateDiscreteInput(z0, 0.0, iuValidate[i])
    zPod.append(zi)
    for j in range(zPod[i].shape[0]):
        alphaPod[i][j, :] = (zPod[i][j, :] - surrogate.modelData.yMean[i][:, 0]) @ surrogate.modelData.M2D @ \
                            surrogate.modelData.Psi[i]
        alphaFull[i][j, :] = (dataSetValidation.rawData.z[i][j, :] - surrogate.modelData.yMean[i][:, 0]) @ \
                             surrogate.modelData.M2D @ surrogate.modelData.Psi[i]

JFull = np.zeros([zPod[i].shape[0], surrogate.uGrid.shape[0]], dtype=float)
JRed = np.zeros([zPod[i].shape[0], surrogate.uGrid.shape[0]], dtype=float)
for j in range(surrogate.uGrid.shape[0]):
    for i in range(zPod[j].shape[0]):
        delta = (zPod[j][i, :] - dataSetValidation.rawData.z[j][i, :]) @ surrogate.modelData.M2D @ (
                    zPod[j][i, :] - dataSetValidation.rawData.z[j][i, :]).T
        JFull[i, j] = (dataSetValidation.rawData.z[j][i, :] - surrogate.modelData.yMean[j][0, :]) @ \
                      surrogate.modelData.M2D @ (dataSetValidation.rawData.z[j][i, :] - surrogate.modelData.yMean[j][0, :]).T
        JRed[i, j] = (zPod[j][i, :] - surrogate.modelData.yMean[j][0, :]) @ surrogate.modelData.M2D @ (
                    zPod[j][i, :] - surrogate.modelData.yMean[j][0, :]).T

# si = 0
# for j in range(3):
#     of.writeField(surrogate.modelData.yMean[j].T, 'U', iStart=197 + j)
#     for i in range(5):
#         of.writeField(zPod[j][i, :], 'U', iStart=200 + si)
#         of.writeField(dataSetValidation.rawData.z[j][i, :], 'U', iStart=200 + si + 1)
#         of.writeField(zPod[j][i, :] - dataSetValidation.rawData.z[j][i, :], 'U', iStart=200 + si + 2)
#         of.writeField(zPod[j][i, :] - surrogate.modelData.yMean[j][0, :], 'U', iStart=200 + si + 3)
#         si += 4

# Compare states and control
plot(z0={'t': dataSetValidation.rawData.t[0], 'z0': alphaFull[0], 'iplot': 0},
     z0r={'t': ti, 'z0r': alphaPod[0], 'iplot': 0},
     z1={'t': dataSetValidation.rawData.t[1], 'z1': alphaFull[1], 'iplot': 1},
     z1r={'t': ti, 'z1r': alphaPod[1], 'iplot': 1},
     z2={'t': dataSetValidation.rawData.t[2], 'z2': alphaFull[2], 'iplot': 2},
     z2r={'t': ti, 'z2r': alphaPod[2], 'iplot': 2},
     J={'t': ti, 'J': JFull, 'iplot': 3},
     Jr={'t': ti, 'Jr': JRed, 'iplot': 3})

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
# zRef = np.zeros([nRef, nModes + 1], dtype=float)
# reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)
reference = ClassReferenceTrajectory(model, T=TRef, zRef=yBase)

# Create class for the MPC problem
MPC = ClassMPC(np=5, nc=1, nch=2, typeOpt='continuous', scipyMinimizeMethod='SLSQP') # scipyMinimizeMethod='trust-constr'

# Weights for the objective function
Q = surrogate.modelData.M2D  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3},
     fOut=pathOut + '/MPC1')

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR_coarse'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5},
     fOut=pathOut + '/MPC2')

print('Done')
