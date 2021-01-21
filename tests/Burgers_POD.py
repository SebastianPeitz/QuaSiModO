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

from QuaSiModO import *
from visualization import *
from models import burgers

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #

T = 5.0
h = 0.005

L = 1.0
dx = 1.0 / 100.0
Re = 100.0
xObs = np.arange(0, L + dx, dx)

uMin = [-1.0, -1.0, -1.0, -1.0, -1.0]
# uMin = [-0.5, -0.5, -0.5, -0.5, -0.5]
uMax = [1.0, 1.0, 1.0, 1.0, 1.0]
# uMin = [-0.025]
# uMax = [0.075]

Ttrain = 50.0  # Time for the simulation in the traing data generation
nLag = 1  # Lag time for EDMD

params = {'Re': Re, 'flagDirichlet0': True}
nModes = 12

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

# Create model class
model = ClassModel('burgers.py', h=h, uMin=uMin, uMax=uMax, params=params, dimZ=len(xObs), typeUGrid='centerStar')
model.setGrid1D(L, dx, xObs)

# y0 = np.linspace(1.0, 0.0, len(model.grid.x))
y0 = np.zeros([len(model.grid.x)], dtype=float)
y0[model.grid.x <= L / 2.0] = 1.0
# y0[model.grid.x > L/2.0] = -1.0
y0[0] = 0.0
y0[-1] = 0.0
# y0 = 1.0 * np.sin(model.grid.x * 2.0 * np.pi / model.grid.x[-1])
z0 = y0[model.grid.iObs]

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=10, nhMax=100)

# Create a data set (and save it to an npz file)
y0Train = list()
y0Train.append(y0)
# y0Train.append(0.0 * np.ones([len(model.grid.x)], dtype=float))
# y0Train.append(np.linspace(1.0, 0.0, len(model.grid.x)))
dataSet.createData(model=model, y0=y0Train, u=uTrain)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='', rawData=dataSet.rawData, nLag=nLag)

# Plot the control u, the control index iu and the corresponding data sets for y and z
# plot(u={'t': dataSet.rawData.t[0], 'u': dataSet.rawData.u[0], 'iplot': 0},
#      iu={'t': dataSet.rawData.t[0], 'iu': dataSet.rawData.iu[0], 'iplot': 1},
#      y={'t': dataSet.rawData.t[0], 'y': dataSet.rawData.y[0], 'iplot': 2, 'legend': False},
#      z={'t': dataSet.rawData.t[0], 'z': dataSet.rawData.z[0], 'iplot': 3})
# plot(y={'t': dataSet.rawData.t[0][::100], 'y': dataSet.rawData.y[0][::100], 'type': 'Surface', 'x': model.grid.x, 'iplot': 0})

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #
Chi_u = burgers.createChi(model)
fig = plt.figure()
for i in range(len(Chi_u)):
    plt.plot(model.grid.x, Chi_u[i])
plt.grid(True)
plt.show()
surrogate = ClassSurrogateModel('POD_burgers.py', uGrid=model.uGrid, h=model.h, dimZ=model.dimZ, z0=z0, nLag=nLag,
                                nModes=nModes, dx=dx, Re=Re, Chi_u=Chi_u)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

# Create comparison data
dataSetValidation = ClassControlDataSet(h=model.h, T=1.0)
uValidate, iuValidate = None, None
for i in range(surrogate.uGrid.shape[0]):
    uValidate, iuValidate = dataSetValidation.createControlSequence(model, typeSequence=surrogate.uGrid[i, :],
                                                                    u=uValidate, iu=iuValidate)

# Create a data set
dataSetValidation.createData(model=model, y0=y0Train, u=uValidate)

# Simulate surrogate model using every nLag'th entry of the training input
zPod = list()
alphaPod = list()
alphaFull = list()

for i in range(surrogate.uGrid.shape[0]):
    alphaPod.append(np.zeros([dataSetValidation.rawData.z[i].shape[0], surrogate.modelData.A[i].shape[0]], dtype=float))
    alphaFull.append(
        np.zeros([dataSetValidation.rawData.z[i].shape[0], surrogate.modelData.A[i].shape[0]], dtype=float))

for i in range(surrogate.uGrid.shape[0]):
    [zi, ti] = surrogate.integrateDiscreteInput(z0, 0.0, iuValidate[i])
    zPod.append(zi)
    for j in range(zPod[i].shape[0]):
        alphaPod[i][j, :] = surrogate.modelData.Psi[i].transpose() @ zPod[i][j, :]
        alphaFull[i][j, :] = surrogate.modelData.Psi[i].transpose() @ dataSetValidation.rawData.z[i][j, :]

# delta = np.zeros([surrogate.uGrid.shape[0], zPod[0].shape[0]], dtype=float)
# for j in range(surrogate.uGrid.shape[0]):
#     for i in range(zPod[j].shape[0]):
#         delta[j, i] = surrogate.modelData.Psi[i].transpose() @ (zPod[j][i, :] - dataSetValidation.rawData.z[j][i, :])

# Compare states and control
plot(z0={'t': dataSetValidation.rawData.t[0], 'z0': alphaFull[0], 'iplot': 0},
     z0r={'t': ti, 'z0r': alphaPod[0], 'iplot': 0},
     z1={'t': dataSetValidation.rawData.t[1], 'z1': alphaFull[1], 'iplot': 1},
     z1r={'t': ti, 'z1r': alphaPod[1], 'iplot': 1},
     z2={'t': dataSetValidation.rawData.t[2], 'z2': alphaFull[2], 'iplot': 2},
     z2r={'t': ti, 'z2r': alphaPod[2], 'iplot': 2},
     z3={'t': dataSetValidation.rawData.t[3], 'z3': alphaFull[3], 'iplot': 3},
     z3r={'t': ti, 'z3r': alphaPod[3], 'iplot': 3})

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1

zRef = 0.0 * np.ones([nRef, 1], dtype=float)
# # zRef[:, 0] = 0.5
# tRef = np.array(np.linspace(0.0, T, nRef))
# zRef[:, 0] = 0.5 + 0.05 * np.sin(np.pi * tRef / 30.0)

# zRef = np.zeros([nRef, len(y0)], dtype=float)
# for i in range(nRef):
#     zRef[i, :] = np.sin(model.grid.x * 2.0 * np.pi / model.grid.x[-1])  # y0

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef)

# Create class for the MPC problem
MPC = ClassMPC(np=5, nc=1, nch=5, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  # , 'trust-constr', 'L-BFGS-B')

# Weights for the objective function
Q = [dx]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# y0 = np.zeros([len(model.grid.x)], dtype=float)
# z0 = y0[model.grid.iObs]

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S, updateSurrogate=True)
resultCont.saveMat('Burgers_POD_Cont', pathOut)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})
plot(y={'t': resultCont.t, 'y': resultCont.y, 'type': 'Surface', 'x': model.grid.x, 'iplot': 0})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S, updateSurrogate=True)
result_SUR.saveMat('Burgers_POD_SUR', pathOut)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})
plot(y={'t': result_SUR.t[::100], 'y': result_SUR.y[::100, :], 'type': 'Surface', 'x': model.grid.x, 'iplot': 0})
