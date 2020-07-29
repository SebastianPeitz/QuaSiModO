from QuaSiModO import *
from visualization import *
import os

import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
T = 20.0  # Time for the MPC problem
h = 0.05  # Time step for the ODE solver, for the training data sampling and for the MPC step length

uMin = [-0.2]  # Minimal value of the input (also defines dimU)
#uMax = [2.0]  # Maximal value of the input (also defines dimU)
#uMax = [1.0]
uMax = [0.5]

nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])
uGrid = np.array([uMin, [0.0], uMax])

Ttrain = 20.0  # Time for the simulation in the traing data generation
nLag = 5  # Lag time for EDMD

tau = 2.0
dimZ = 5

y0 = list()
nSpline = 4
tSpline = np.linspace(-tau, 0.0, nSpline)
tTau = np.linspace(-tau, 0.0, int(tau/h) + 1)
tck = interpolate.splrep(tSpline, 0.5 + 2.0 * np.random.rand(nSpline), s=0)
y0.append(interpolate.splev(tTau, tck, der=0))
params = {'tau': tau}

#pathData = 'tests/results/MackeyGlass/data_ESN_2_0'
#pathData = 'tests/results/MackeyGlass/data_ESN_1_0'
pathData = 'tests/results/MackeyGlass/v3/data_ESN_0_5'




approx_res_size = 130# 2000 #1500
radius = 0.9 #0.9
sparsity = 0.9 # 0.7 #0.8


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

uTrain, iuTrain = dataSet.createControlSequence(model, T=1000.0, typeSequence='piecewiseConstant', nhMin=5, nhMax=5)#, u=uTrain,iu=iuTrain)#,u=uTrain,iu=iuTrain)

# Create a data set (and save it to an npz file)

if os.path.exists(pathData + '.npz'):
    dataSet.createData(loadPath=pathData)
else:
    dataSet.createData(model=model, y0=y0, u=uTrain, savePath=pathData)


#plotPhase3D(dataSet.rawData.z[0][:, 0], dataSet.rawData.z[0][:, 2], dataSet.rawData.z[0][:, 4])

y0 = dataSet.rawData.y[-1][-1, :]
z0 = dataSet.rawData.z[-1][-1, :]

# For ESN we do not need data preparation. We take the mixed raw data

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

surrogate = ClassSurrogateModel('ESN.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ,
                                z0=z0, nDelay=0, nLag=nLag,
                                approx_res_size=approx_res_size,spectral_radius = radius, sparsity=sparsity)

surrogate.createROM(dataSet.rawData)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare surrogate model with full model
# -------------------------------------------------------------------------------------------------------------------- #

# # Simulate surrogate model using every nLag'th entry of the training input
Tcomp = 5.0
tS = np.linspace(0.0, Tcomp, int(Tcomp / (nLag * model.h)) + 1)
tF = np.linspace(0.0, Tcomp, int(Tcomp / (model.h)) + 1)
zS = list()
zF = list()
for i in range(model.nU):

    iu = i * np.ones([len(tS), 1], dtype=int)
    [z, tSurrogate] = surrogate.integrateDiscreteInput(z0, 0.0, iu)
    zS.append(z)

    u = model.uGrid[i, 0] * np.ones([len(tF), 1], dtype=int)
    [yOpt, zOpt, tOpt, model] = model.integrate(y0, u, 0.0)
    zF.append(zOpt)
    
iu = iuTrain[-1][:len(tF):nLag,:]    
[z, tSurrogate] = surrogate.integrateDiscreteInput(z0, 0.0, iu)
zS.append(z)

u = uTrain[-1][:len(tF),:]
[yOpt, zOpt, tOpt, model] = model.integrate(y0, u, 0.0)
zF.append(zOpt)

# Compare states and control
plot(z0={'t': tOpt, 'z0': zF[0], 'iplot': 0},
      z0r={'t': tSurrogate, 'z0r': zS[0], 'iplot': 0},
      z1={'t': tOpt, 'z1': zF[1], 'iplot': 1},
      z1r={'t': tSurrogate, 'z1r': zS[1], 'iplot': 1},
      z2={'t': tOpt, 'z2': zF[2], 'iplot': 2},
      z2r={'t': tSurrogate, 'z2r': zS[2], 'iplot': 2},
      z3={'t': tOpt, 'z3': zF[3], 'iplot': 3},
      z3r={'t': tSurrogate, 'z3r': zS[3], 'iplot': 3}
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
zRef[:, 0] = 1.0 #+ 0.5 * np.sin(2.0 * tRef * 2.0 * np.pi / TRef)

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef)

scipyMinimizeOptions = {'epsilon': 1e-10}

# Create class for the MPC problem
MPC = ClassMPC(np=5, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP', scipyMinimizeOptions=scipyMinimizeOptions)  # scipyMinimizeMethod=

# Weights for the objective function
Q = [1.0, 0.0, 0.0, 0.0, 0.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

save_path_cont = 'Figures_Paper/MackeyGlass_ESN/Ctrl_0_5/MackeyGlass_ESN_cont_0_5'
save_path_SUR = 'Figures_Paper/MackeyGlass_ESN/Ctrl_0_5/MackeyGlass_ESN_SUR_0_5'

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, z0=z0, T=T, Q=Q, R=R, S=S,savePath=save_path_cont)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
      u={'t': resultCont.t[:-1], 'u': resultCont.u, 'iplot': 1},
      J={'t': resultCont.t[:-1], 'J': resultCont.J, 'iplot': 2},
      nFev={'t': resultCont.t[:-1], 'nFev': resultCont.nFev, 'iplot': 3})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, z0=z0, T=T, Q=Q, R=R, S=S,  savePath=save_path_SUR)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
      u={'t': result_SUR.t[:-1], 'u': result_SUR.u, 'iplot': 1},
      J={'t': result_SUR.t[:-1], 'J': result_SUR.J, 'iplot': 2},
      nFev={'t': result_SUR.t[:-1], 'nFev': result_SUR.nFev, 'iplot': 3},
      alpha={'t': result_SUR.t[:-1], 'alpha': result_SUR.alpha, 'iplot': 4},
      omega={'t': result_SUR.t[:-1], 'omega': result_SUR.omega, 'iplot': 5})
