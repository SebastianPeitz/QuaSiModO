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
T = 5.0  # Time for the MPC problem
Ttrain = 100.0  # Time for the simulation in the traing data generation
h = 0.1  # Time step for the ODE solver, for the training data sampling and for the MPC step length
y0 = [1.0, 2.0]  # Initial condition of the ODE
uMin = [-2.0]  # Minimal value of the input (also defines dimU)
uMax = [2.0]  # Maximal value of the input (also defines dimU)

typeSurrogate = 'EDMD'  #'validityCheck'  #
nLag = 5  # Lag time for EDMD
nMonomials = 2  # Max order of monomials for EDMD

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

model = ClassModel(rhs, h=h, uMin=uMin, uMax=uMax, dimZ=2, typeUGrid='cube', nGridU=2)

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=10, nhMax=20)

# Create a data set (and save it to an npz file)
dataSet.createData(model=model, y0=y0, u=uTrain, savePath='tests/results/TestODE_Data')

# Load the data set that was created using the loadPath
dataSet.createData(loadPath='tests/results/TestODE_Data')

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y_dX', rawData=dataSet.rawData, nLag=nLag)

# Plot the control u, the control index iu and the corresponding data sets for y and z
# multiPlotLines2D(dataSet.rawData.t[0], u=dataSet.rawData.u[0], iu=dataSet.rawData.iu[0],
#                  y=dataSet.rawData.y[0], z=dataSet.rawData.z[0])
plot(u={'t': dataSet.rawData.t[0], 'u': dataSet.rawData.u[0], 'iplot': 0},
          iu={'t': dataSet.rawData.t[0], 'iu': dataSet.rawData.iu[0], 'iplot': 1},
          y={'t': dataSet.rawData.t[0], 'y': dataSet.rawData.y[0], 'iplot': 2},
          z={'t': dataSet.rawData.t[0], 'z': dataSet.rawData.z[0], 'iplot': 3})

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

# Create class for the surrogate model
# a) validityCheck: Surrogate model is equal to the original model, but restricted to inputs contained in uGrid
if typeSurrogate == 'validityCheck':
    surrogate = ClassSurrogateModel('ODEToySurrogate.py', uGrid=model.uGrid, h=model.h, dimZ=model.dimZ, z0=y0)

# b) EDMD
elif typeSurrogate == 'EDMD':
    surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ, z0=y0,
                                    nLag=nLag, nMonomials=nMonomials)

surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

# Create random input sequence on which the input is constant over nLag time steps
nCoarse = int(Ttrain/surrogate.h) + 1
tCoarse = np.linspace(0.0, Ttrain, nCoarse)
iuCoarse = np.zeros([nCoarse, 1], dtype=int)
iuCoarse[:, 0] = np.random.randint(0, model.nU - 1, nCoarse)
uFine = np.zeros([int(Ttrain/model.h) + 1, 1], dtype=float)
for i in range(nCoarse):
    uFine[i * nLag: (i + 1) * nLag, 0] = model.uGrid[iuCoarse[i], :]

# Simulate system using the model.integrate call with random control u and initial time t0 = 0.0
[y, z, t, _] = model.integrate(y0, uFine, 0.0)

# Simulate surrogate model using every nLag'th entry of the training input
[z, tSurrogate] = surrogate.integrateDiscreteInput(y0, 0.0, iuCoarse)

# Compare states and control
plot(y={'t': t, 'y': y, 'iplot': 0}, z={'t': tSurrogate, 'z': z, 'markerSize': 5, 'iplot': 0},
     uF={'t': t, 'uF': uFine, 'iplot': 1}, uS={'t': tSurrogate, 'uS': model.uGrid[iuCoarse, :], 'markerSize': 5,
                                               'iplot': 1})

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Create class for the MPC problem
MPC = ClassMPC(np=3, nc=1, typeOpt='continuous', scipyMinimizeMethod='trust-constr')

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 2.0
zRef = 0.5 * np.ones([int(round(TRef / h)) + 1, 1], dtype=float)
iRef = [1]

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)

# Weights for the objective function
Q = [0.0, 1.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# # 1) Full model, continuous control
# MPC.nch = nLag
# result_FullContinuous = MPC.run(model, reference, y0=y0, T=T, Q=Q, R=R, S=S)
#
# plot(z={'t': result_FullContinuous.t, 'z': result_FullContinuous.z, 'reference': reference, 'iplot': 0},
#      u={'t': result_FullContinuous.t[:-1], 'u': result_FullContinuous.u, 'iplot': 1},
#      J={'t': result_FullContinuous.t[:-1], 'J': result_FullContinuous.J, 'iplot': 2},
#      nFev={'t': result_FullContinuous.t[:-1], 'nFev': result_FullContinuous.nFev, 'iplot': 3})
#
# # 2) Full model, integer control, computed via total evaluation of all control combinations on the control horizon
# MPC.typeOpt = 'combinatorial'
# result_FullCombinatorial = MPC.run(model, reference, y0=y0, T=T, Q=Q, R=R, S=S)
#
# plot(z={'t': result_FullCombinatorial.t, 'z': result_FullCombinatorial.z, 'reference': reference, 'iplot': 0},
#      u={'t': result_FullCombinatorial.t[:-1], 'u': result_FullCombinatorial.u, 'iplot': 1},
#      J={'t': result_FullCombinatorial.t[:-1], 'J': result_FullCombinatorial.J, 'iplot': 2},
#      nFev={'t': result_FullCombinatorial.t[:-1], 'nFev': result_FullCombinatorial.nFev, 'iplot': 3})

# 3) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
MPC.typeOpt = 'continuous'
MPC.nch = 1
result_SurrogateContinuous = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SurrogateContinuous.t, 'z': result_SurrogateContinuous.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SurrogateContinuous.t, 'u': result_SurrogateContinuous.u, 'iplot': 1},
     J={'t': result_SurrogateContinuous.t, 'J': result_SurrogateContinuous.J, 'iplot': 2},
     nFev={'t': result_SurrogateContinuous.t, 'nFev': result_SurrogateContinuous.nFev, 'iplot': 3})

# 4) Surrogate model, integer control computed via total evaluation of all control combinations on the control horizon
MPC.typeOpt = 'combinatorial'
result_SurrogateCombinatorial = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SurrogateCombinatorial.t, 'z': result_SurrogateCombinatorial.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SurrogateCombinatorial.t, 'u': result_SurrogateCombinatorial.u, 'iplot': 1},
     J={'t': result_SurrogateCombinatorial.t, 'J': result_SurrogateCombinatorial.J, 'iplot': 2},
     nFev={'t': result_SurrogateCombinatorial.t, 'nFev': result_SurrogateCombinatorial.nFev, 'iplot': 3})

# 5) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR'
result_SurrogateSUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SurrogateSUR.t, 'z': result_SurrogateSUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SurrogateSUR.t, 'u': result_SurrogateSUR.u, 'iplot': 1},
     J={'t': result_SurrogateSUR.t, 'J': result_SurrogateSUR.J, 'iplot': 2},
     nFev={'t': result_SurrogateSUR.t, 'nFev': result_SurrogateSUR.nFev, 'iplot': 3},
     alpha={'t': result_SurrogateSUR.t, 'alpha': result_SurrogateSUR.alpha, 'iplot': 4},
     omega={'t': result_SurrogateSUR.t, 'omega': result_SurrogateSUR.omega, 'iplot': 5})

# 6) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR_coarse'
result_SurrogateSURc = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SurrogateSURc.t, 'z': result_SurrogateSURc.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SurrogateSURc.t, 'u': result_SurrogateSURc.u, 'iplot': 1},
     J={'t': result_SurrogateSURc.t, 'J': result_SurrogateSURc.J, 'iplot': 2},
     nFev={'t': result_SurrogateSURc.t, 'nFev': result_SurrogateSURc.nFev, 'iplot': 3},
     alpha={'t': result_SurrogateSURc.t, 'alpha': result_SurrogateSURc.alpha, 'iplot': 4},
     omega={'t': result_SurrogateSURc.t, 'omega': result_SurrogateSURc.omega, 'iplot': 5})

# -------------------------------------------------------------------------------------------------------------------- #
# Store the final result to a .mat file
# -------------------------------------------------------------------------------------------------------------------- #
result_SurrogateSUR.saveMat("tests/results/TestODE_MPC_SUR_Surrogate.mat")
