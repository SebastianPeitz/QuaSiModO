from QuaSiModO import *
from visualization import *

import numpy as np
import matplotlib.pyplot as plt

# %%

# -------------------------------------------------------------------------------------------------------------------- #
# Choose method and set parameters
# -------------------------------------------------------------------------------------------------------------------- #

method = "ESN"

sigma = 10.0 # Paramters for Lorenz system
r = 28.0
b = 8/3


T = 10.0 # Time for the MPC problem
h = 0.01  # Time step for the ODE solver, for the training data sampling and for the MPC step length
y0 = [0.5,-0.5,1.0]  # Initial condition of the ODE
uMin = [-50.0]  # Minimal value of the input (also defines dimU)
uMax = [50.0]  # Maximal value of the input (also defines dimU)

dimZ = 3  # dimension of the observable (= dimY in the ODE case, unless iObs is passed to ClassModel)
nGridU = 1  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])

Ttrain = 1000  # Time for the simulation in the traing data generation

if method == "EDMD":
    nLag = 5  # Lag time for EDMD
    nMonomials = 3  # Max order of monomials for EDMD
    nDelay = 0 # Number of Delay steps for EDMD (EDMD seems to work worse with delay)
elif method == "LSTM":
    nDelay = 2 # Number of Delay steps for LSTM (LSTM needs a delay)
    nLag = 5 # Lag time for LSTM
    nhidden = 15 # number of hidden neurons in LSTM cell
elif method == "ESN":
    nDelay = 0
    nLag = 5 # Lag time for LSTM
    approx_res_size = 250
    radius = 0.9
    sparsity = 0.5



# %%
    
# -------------------------------------------------------------------------------------------------------------------- #
# ODE: Define right-hand side as a a function of the state y and the control u
# -------------------------------------------------------------------------------------------------------------------- #
def rhs(y_, u_):
    
    y = np.zeros([3,]);

    y[0] = sigma * (y_[1] - y_[0]);
    y[1] = r * y_[0] - y_[0] * y_[2] - y_[1] + u_;
    y[2] = y_[0] * y_[1] - b * y_[2];    
    
    return y

# %%

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

# Create model class
model = ClassModel(rhs, h=h, uMin=uMin, uMax=uMax, dimZ=dimZ, typeUGrid='cube', nGridU=nGridU)

# %%

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=5, nhMax=20)
uTrain, iuTrain = dataSet.createControlSequence(model, h = model.h, T=500.0, typeSequence=[0.0], nhMin=10, nhMax=10, u=uTrain,iu=iuTrain)

if method == "ESN":
    uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=10, nhMax=10)

# Create a data set (and save it to an npz file)
dataSet.createData(model=model, y0=y0, u=uTrain)

# prepare data according to the desired reduction scheme

data = dataSet.prepareData(model, method='Y', rawData=dataSet.rawData, nLag=nLag, nDelay=nDelay)

# %%

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

if method == "EDMD":

    surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, 
                                    h=nLag * model.h, dimZ=model.dimZ, 
                                    z0=y0, nDelay=nDelay, nLag=nLag, 
                                    nMonomials=nMonomials, epsUpdate=0.05)
    
elif method == "LSTM":
    
    surrogate = ClassSurrogateModel('LSTM.py', uGrid=model.uGrid, 
                                    h=nLag * model.h, dimZ=model.dimZ, 
                                    z0=y0, nDelay=nDelay, nhidden=nhidden) 
    
elif method == "ESN":
    
    surrogate = ClassSurrogateModel('ESN.py', uGrid=model.uGrid, 
                                    h=nLag * model.h, dimZ=model.dimZ, 
                                    z0=y0, nDelay=0, nLag=nLag,
                                    approx_res_size=approx_res_size,spectral_radius = radius, sparsity=sparsity) 
    
surrogate.createROM(dataSet.rawData)
#surrogate.createROM(data)
    
# %% 

# -------------------------------------------------------------------------------------------------------------------- #
# Test surrogate model
# -------------------------------------------------------------------------------------------------------------------- #

start = np.random.randint(nDelay + 1, dataSet.rawData.z[0].shape[0])
steps = 1000

iu = np.random.randint(0,nGridU + 1,[int(steps/nLag),1])
u = np.zeros([steps,1])
for i in range(int(steps/nLag)):
    u[nLag * i :nLag * (i+1),0] = model.uGrid[iu[i,0]]


y0 = dataSet.rawData.z[0][start,:]

temp = dataSet.rawData.z[0][start:start-nDelay*nLag-1:-nLag,:]
z0 = np.reshape(temp, [dimZ*(nDelay+1)])

if method == "ESN":

    y0 = dataSet.rawData.y[-1][-1, :]
    z0 = dataSet.rawData.z[-1][-1, :]

y_model, t_model = surrogate.integrateDiscreteInput(z0, 0.0, iu)
y_ode,_, t_ode,_ = model.integrate(y0,u,0.0)

    
fig, axs = plt.subplots(nrows = 3, ncols=1, constrained_layout=True,figsize=(10, 6))
plt.title("Test performance", fontsize=12)
for i in range(3):

    axs[i].plot(t_model, y_model[:,i], linewidth=2)
    axs[i].plot(t_ode, y_ode[:,i], linewidth=2)
    axs[i].set_ylabel(r"$x[$" + str(i) + r"$]$", fontsize=12)

plt.xlabel(r"$t$", fontsize=12) 

# plt.savefig("Figures/test_prediction_surrogate.pdf", bbox_inches="tight")
    
plt.show()
 
# %%

# -------------------------------------------------------------------------------------------------------------------- #
# Initilize MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
iRef = [1]

TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 1], dtype=float)

## Sinus trajectory:
tRef = np.array(np.linspace(0.0, T, nRef))
zRef[:, 0] = 1.5 * np.sin(2.0 * tRef * 2.0 * np.pi / TRef)

## Piecewise constant
# zRef[:, 0] = 2.0
# tRef = np.array(np.linspace(0.0, T, nRef))
# zRef[np.where(tRef >= 10.0/10*2), 0] = 1.0
# zRef[np.where(tRef >= 20.0/10*2), 0] = -1.0
# zRef[np.where(tRef >= 30.0/10*2), 0] = 0.5
# zRef[np.where(tRef >= 40.0/10*2), 0] = -1.0

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)


# Create class for the MPC problem
MPC = ClassMPC(np=3, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  # scipyMinimizeMethod=


# Weights for the objective function
Q = [0.0, 1.0, 0.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# %%

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

print("shapes:")

plot(z={'t': resultCont.t, 'z': np.reshape(resultCont.z[:,1],[resultCont.z.shape[0],1]), 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})

# # 2) Surrogate model, integer control computed via relaxation and sum up rounding
# MPC.typeOpt = 'SUR'
# result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

# plot(z={'t': result_SUR.t, 'z': np.reshape(result_SUR.z[:,1],[result_SUR.z.shape[0],1]), 'reference': reference, 'iplot': 0},
#      u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
#      J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
#      nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
#      alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
#      omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})
