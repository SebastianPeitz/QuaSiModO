from QuaSiModO import *
from visualization import *

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
T = 60.0
h = 0.005

L = 2.0
dx = 1.0 / 24.0
Re = 100.0
xObs = [0.0, 0.5, 1.0, 1.5]

uMin = [-0.05]
uMax = [0.075]

Ttrain = 100.0  # Time for the simulation in the traing data generation
nLag = 100  # Lag time for EDMD
nMonomials = 3  # Max order of monomials for EDMD

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

# Create model class
model = ClassModel('burgers.py', h=h, uMin=uMin, uMax=uMax, Re=Re, dimZ=len(xObs), typeUGrid='oneSidedStar')
model.setGrid1D(L, dx, xObs)

y0 = np.linspace(0.0, 1.0, len(model.grid.x))
z0 = y0[model.grid.iObs]

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=100, nhMax=500)

# Create a data set (and save it to an npz file)
y0Train = list()
y0Train.append(y0)
y0Train.append(np.linspace(1.0, 0.0, len(model.grid.x)))
dataSet.createData(model=model, y0=y0Train, u=uTrain)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y', rawData=dataSet.rawData, nLag=nLag)

# Plot the control u, the control index iu and the corresponding data sets for y and z
plot(u={'t': dataSet.rawData.t[0], 'u': dataSet.rawData.u[0], 'iplot': 0},
     iu={'t': dataSet.rawData.t[0], 'iu': dataSet.rawData.iu[0], 'iplot': 1},
     y={'t': dataSet.rawData.t[0], 'y': dataSet.rawData.y[0], 'iplot': 2, 'legend': False},
     z={'t': dataSet.rawData.t[0], 'z': dataSet.rawData.z[0], 'iplot': 3})

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ, z0=z0, nLag=nLag, nMonomials=nMonomials, epsUpdate=0.05)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 2.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 1], dtype=float)

# zRef[:, 0] = 3.0
tRef = np.array(np.linspace(0.0, T, nRef))
zRef[:, 0] = 0.5 + 0.05 * np.sin(np.pi * tRef / 30.0)

iRef = [0, 1, 2, 3]

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef, iRef=iRef)

# Create class for the MPC problem
MPC = ClassMPC(np=5, nc=1, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  #, 'trust-constr', 'L-BFGS-B')

# Weights for the objective function
Q = [1.0, 1.0, 1.0, 1.0]  # reference tracking: (z - deltaZ)^T * Q * (z - deltaZ)
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'reference': reference, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})
plot(y={'t': resultCont.t[::100], 'y': resultCont.y[::100, :], 'type': 'Surface', 'x': model.grid.x, 'iplot': 0})

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'reference': reference, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})
plot(y={'t': result_SUR.t[::100], 'y': result_SUR.y[::100, :], 'type': 'Surface', 'x': model.grid.x, 'iplot': 0})
