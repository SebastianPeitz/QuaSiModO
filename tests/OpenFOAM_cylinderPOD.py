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

nInputs = 1
dimInputs = 1
iInputs = [0]

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define what to observe
# -------------------------------------------------------------------------------------------------------------------- #

nModes = 15

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
Re = 100.0
hFOAM = 0.01
dimSpace = 2

T = 60.0
h = 0.1

uMin = [-2.0]
uMax = [2.0]
nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])

Ttrain = 60.0  # Time for the simulation in the traing data generation

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

obs = ClassObservable(flagFullState=True)

of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, typeUGrid='cube', nGridU=nGridU)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)
dimZ = dataSet.dimZ
model.dimZ = dimZ

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=5, nhMax=20)
# uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=[0.0], u=uTrain, iu=iuTrain)

# Create a data set (and save it to an npz file)
if os.path.exists(pathData + '.npz'):
    dataSet.createData(loadPath=pathData)
else:
    dataSet.createData(model=model, u=uTrain, savePath=pathData)

# plot data
t = np.linspace(0.0, Ttrain, uTrain[0].shape[0])
plot(u0={'t': dataSet.rawData.t[0], 'u0': dataSet.rawData.u[0], 'iplot': 0})

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='', rawData=dataSet.rawData)

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

yBase, _ = of.readSolution(999.0, 999.0, 1.0, whichField='U')
yC, _ = of.readSolution(899.0, 899.0, 1.0, whichField='U')

z0 = dataSet.rawData.z[0][0, :]
surrogate = ClassSurrogateModel('POD.py', uGrid=model.uGrid, h=model.h, dimZ=z0.shape[0], z0=z0,
                                yBase=yBase.T, yC=yC.T, nModes=nModes, of=of, Re=Re)
if os.path.exists(pathROM):
    surrogate.createROM(data, loadPath=pathROM)
else:
    surrogate.createROM(data, savePath=pathROM)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

pathDataValidation = pathOut + '/dataValidation'

# Create comparison data
dataSetValidation = ClassControlDataSet(h=model.h, T=20.0)
uValidate, iuValidate = dataSetValidation.createControlSequence(model, typeSequence=[0.0])
uValidate, iuValidate = dataSetValidation.createControlSequence(model, typeSequence=[2.0], u=uValidate, iu=iuValidate)
uValidate, iuValidate = dataSetValidation.createControlSequence(model, typeSequence=[-2.0], u=uValidate, iu=iuValidate)
mapValidateToUGrid = [2, 0, 1]

# Create a data set (and save it to an npz file)
if os.path.exists(pathDataValidation + '.npz'):
    dataSetValidation.createData(loadPath=pathDataValidation)
else:
    dataSetValidation.createData(model=model, u=uValidate, savePath=pathDataValidation)

# Simulate surrogate model using every nLag'th entry of the training input
zPod = list()
alphaPod = list()
alphaFull = list()

for i in range(3):
    alphaPod.append(np.zeros([dataSetValidation.rawData.z[i].shape[0], surrogate.modelData.A[i].shape[0]], dtype=float))
    alphaFull.append(np.zeros([dataSetValidation.rawData.z[i].shape[0], surrogate.modelData.A[i].shape[0]], dtype=float))

for i in range(3):
    [zi, ti] = surrogate.integrateDiscreteInput(z0, 0.0, iuValidate[i])
    zPod.append(zi)
    for j in range(zPod[i].shape[0]):
        alphaPod[i][j, :] = (zPod[i][j, :] - surrogate.modelData.yMean[i][:, 0]) @ surrogate.modelData.M2D @ \
                            surrogate.modelData.Psi[i]
        alphaFull[mapValidateToUGrid[i]][j, :] = (dataSetValidation.rawData.z[mapValidateToUGrid[i]][j, :] -
                                                  surrogate.modelData.yMean[mapValidateToUGrid[i]][:, 0]) @ \
                             surrogate.modelData.M2D @ surrogate.modelData.Psi[mapValidateToUGrid[i]]

# Compare states and control
plot(z0={'t': dataSetValidation.rawData.t[0], 'z0': alphaFull[0], 'iplot': 0},
     z0r={'t': ti, 'z0r': alphaPod[0], 'iplot': 0},
     z1={'t': dataSetValidation.rawData.t[1], 'z1': alphaFull[1], 'iplot': 1},
     z1r={'t': ti, 'z1r': alphaPod[1], 'iplot': 1},
     z2={'t': dataSetValidation.rawData.t[2], 'z2': alphaFull[2], 'iplot': 2},
     z2r={'t': ti, 'z2r': alphaPod[2], 'iplot': 2})

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 1], dtype=float)

# zRef[:, 0] = 3.0
tRef = np.array(np.linspace(0.0, T, nRef))
zRef[:, 0] = 0.0 + 1.5 * np.sin(2.0 * tRef * 2.0 * np.pi / TRef)

iRef = [1]

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)

# Create class for the MPC problem
MPC = ClassMPC(np=3, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP') # scipyMinimizeMethod='trust-constr'

# Weights for the objective function
Q = np.ones([nModes + 1], dtype=float)  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
# resultCont = MPC.run(model, reference, surrogateModel=surrogate, T=T, Q=Q, R=R, S=S)
#
# plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
#      u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
#      J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
#      nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3},
#      fOut=pathOut + '/MPC1')

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
