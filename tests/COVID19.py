from QuaSiModO import *
from visualization import *
from scipy.io import savemat

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
T = 700.0  # Time for the MPC problem
h = 1.0 / 12.0  # Time step for the ODE solver, for the training data sampling and for the MPC step length
uMin = [0.0]  # Minimal value of the input (also defines dimU)
uMax = [1.0]  # Maximal value of the input (also defines dimU)

Ntotal = 8.3e7

y0 = 1 / Ntotal * np.array([82636256.0, 20581.0, 0.0, 8041.0, 41931.0, 11469.0, 276911.0, 4810.0])

dimZ = 5  # dimension of the observable (= dimY in the ODE case, unless iObs is passed to ClassModel)
uGrid = np.array([[0.0], [0.58216], [0.7062], [1.0]])

Ttrain = 200.0  # Time for the simulation in the traing data generation
nLag = int(1.0 / h)  # Lag time for EDMD
nMonomials = 0  # Max order of monomials for EDMD

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

# Create model class
model = ClassModel('modelCOVID19.py', h=h, uMin=uMin, uMax=uMax, dimZ=dimZ, uGrid=uGrid, typeUGrid='cube')

t = np.linspace(0.0, T, int(T/h) + 1)
u = np.zeros([len(t), 1], dtype=float)

[y, z, t, _] = model.integrate(y0, u, 0.0)
savemat('tests/results/COVID19-ucontrolled.mat', {'y': y, 't': t, 'u': u})

z0 = z[0, :]

# plot(S={'t': t, 'S': y[:, 0], 'iplot': 0},
#      I={'t': t, 'I': y[:, 1], 'iplot': 1},
#      D={'t': t, 'D': y[:, 2], 'iplot': 2},
#      A={'t': t, 'A': y[:, 3], 'iplot': 3},
#      R={'t': t, 'R': y[:, 4], 'iplot': 4},
#      T={'t': t, 'T': y[:, 5], 'iplot': 5},
#      H={'t': t, 'H': y[:, 6], 'iplot': 6},
#      E={'t': t, 'E': y[:, 7], 'iplot': 7})

# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #

# Create data set class
dataSet = ClassControlDataSet(h=model.h, T=Ttrain)

# Create a sequence of controls
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=uGrid[0], nhMin=7 * nLag, nhMax=28 * nLag)
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=uGrid[1], nhMin=7 * nLag, nhMax=28 * nLag, u=uTrain, iu=iuTrain)
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=uGrid[2], nhMin=7 * nLag, nhMax=28 * nLag, u=uTrain, iu=iuTrain)
uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence=uGrid[3], nhMin=7 * nLag, nhMax=28 * nLag, u=uTrain, iu=iuTrain)

yTrain = list()
yTrain.append(y0)
y1 = np.zeros(y0.shape)
for i in range(10):
    y1[1:] = 1e-2 * np.random.rand(7)
    y1[0] = 1.0 - np.sum(y1[1:])
    yTrain.append(y1)
# yTrain.append(y[12 * 25, :])
# yTrain.append(y[12 * 50, :])
# yTrain.append(y[12 * 100, :])

dataSet.createData(model=model, y0=yTrain, u=uTrain)

# prepare data according to the desired reduction scheme
data = dataSet.prepareData(model, method='Y_dX', rawData=dataSet.rawData, nLag=nLag, nDelay=0)

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate modeling
# -------------------------------------------------------------------------------------------------------------------- #

surrogate = ClassSurrogateModel('EDMD.py', uGrid=model.uGrid, h=nLag * model.h, dimZ=model.dimZ, z0=z0, nDelay=0,
                                nLag=nLag, nMonomials=nMonomials, epsUpdate=0.01)
surrogate.createROM(data)

# -------------------------------------------------------------------------------------------------------------------- #
# MPC
# -------------------------------------------------------------------------------------------------------------------- #

# Define reference trajectory for second state variable (iRef = 1)
TRef = T + 200.0
nRef = int(round(TRef / h)) + 1
zRef = np.zeros([nRef, 8], dtype=float)

reference = ClassReferenceTrajectory(model, T=TRef, zRef=zRef)

# Create class for the MPC problem
MPC = ClassMPC(np=4, nc=1, nch=7, typeOpt='continuous', scipyMinimizeMethod='SLSQP')  # scipyMinimizeMethod=

# Weights for the objective function
Q = [0.0]
L = [0.0]
R = [0.0]  # control cost: u^T * R * u
S = [0.0]  # weighting of (u_k - u_{k-1})^T * S * (u_k - u_{k-1})

# -------------------------------------------------------------------------------------------------------------------- #
# Solve different MPC problems (via "MPC.run") and plot the result
# -------------------------------------------------------------------------------------------------------------------- #

# 1) Surrogate model, continuous input obtained via relaxation of the integer input in uGrid
resultCont = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, L=L, R=R, S=S, updateSurrogate=False)

plot(z={'t': resultCont.t, 'z': resultCont.z, 'iplot': 0},
     u={'t': resultCont.t, 'u': resultCont.u, 'iplot': 1},
     J={'t': resultCont.t, 'J': resultCont.J, 'iplot': 2},
     nFev={'t': resultCont.t, 'nFev': resultCont.nFev, 'iplot': 3})
resultCont.saveMat('tests/results/COVID19-cont.mat')

# 2) Surrogate model, integer control computed via relaxation and sum up rounding
MPC.typeOpt = 'SUR_coarse'
result_SUR = MPC.run(model, reference, surrogateModel=surrogate, y0=y0, T=T, Q=Q, R=R, S=S, updateSurrogate=True)

plot(z={'t': result_SUR.t, 'z': result_SUR.z, 'iplot': 0},
     u={'t': result_SUR.t, 'u': result_SUR.u, 'iplot': 1},
     J={'t': result_SUR.t, 'J': result_SUR.J, 'iplot': 2},
     nFev={'t': result_SUR.t, 'nFev': result_SUR.nFev, 'iplot': 3},
     alpha={'t': result_SUR.t, 'alpha': result_SUR.alpha, 'iplot': 4},
     omega={'t': result_SUR.t, 'omega': result_SUR.omega, 'iplot': 5})
result_SUR.saveMat('tests/results/COVID19-SUR.mat')
