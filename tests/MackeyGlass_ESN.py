from sys import path
from os import getcwd, sep
path.append(getcwd()[:getcwd().rfind(sep)])


from QuaSiModO import *
from visualization import *
import os

import pickle

import numpy as np

from scipy.io import savemat

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
T = 20.0  # Time for the MPC problem
h = 0.05  # Time step for the ODE solver, for the training data sampling and for the MPC step length

uMin = [-0.2]  # Minimal value of the input (also defines dimU)
uMax = [1.0]

nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])
uGrid = np.array([uMin, [0.0], uMax])


Ttrain = 20.0  # Time for the simulation in the traing data generation
nLag = 5  # Lag time for EDMD

tau = 2.0
dimZ = 1

y0 = list()
nSpline = 4
tSpline = np.linspace(-tau, 0.0, nSpline)
tTau = np.linspace(-tau, 0.0, int(tau/h) + 1)
tck = interpolate.splrep(tSpline, 0.5 + 2.0 * np.random.rand(nSpline), s=0)
y0.append(interpolate.splev(tTau, tck, der=0))
params = {'tau': tau}


pathData = 'tests/results/MackeyGlass/data_ESN_1_0'
savePath_mat = 'tests/results/MackeyGlass/result_ESN_1_0.mat'

approx_res_size = 200
radius = 0.75 
sparsity = 0.9 


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
    

y0 = dataSet.rawData.y[-1][-nLag, :]
z0 = dataSet.rawData.z[-1][-nLag, :]


# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

surrogate = ClassSurrogateModel('ESN.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ,
                                z0=z0, nDelay=0, nLag=nLag, 
                                approx_res_size=approx_res_size,spectral_radius = radius, sparsity=sparsity)


# For the ESN we need the rawData (not prepared)
# ToDo: Passendes prepareDate schreiben
surrogate.createROM(dataSet.rawData)


# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 5.0
nRef = int(round(TRef / h)) + 1

zRef = np.zeros([nRef, 1], dtype=float)
zRef[:, 0] = 1.0 

tRef = np.array(np.linspace(0.0, T, nRef))

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef)

scipyMinimizeOptions = {'epsilon': 1e-10}

# Create class for the MPC problem
MPC = ClassMPC(np=5, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP', scipyMinimizeOptions=scipyMinimizeOptions)  # scipyMinimizeMethod=

# Weights for the objective function
Q = [1.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

save_path_cont = 'tests/results/MackeyGlass/MackeyGlass_ESN_cont_1_0'
save_path_SUR = 'tests/results/MackeyGlass/MackeyGlass_ESN_SUR_1_0'

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, z0=z0, T=T, Q=Q, R=R, S=S,savePath=save_path_cont)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
      u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
      J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
      nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})


# Not relevant since the control enters lineary
# 2) Surrogate model, integer control computed via relaxation and sum up rounding
# MPC.typeOpt = 'SUR'
# result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, z0=z0, T=T, Q=Q, R=R, S=S,  savePath=save_path_SUR)

# plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
#       u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
#       J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
#       nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
#       alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
#       omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})

savemat(savePath_mat, {'t': resultCont.t, 'z': resultCont.z, 'u': resultCont.u, 'J': resultCont.J, 'nFev': resultCont.nFev, 'zRef': zRef})