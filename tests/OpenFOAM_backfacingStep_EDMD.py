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

from OpenFOAM.classesOpenFOAM import *
from visualization import *

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define model
# -------------------------------------------------------------------------------------------------------------------- #
problem = 'backFacingStep'
nProc = 1

nInputs = 1
dimInputs = 3
iInputs = [0]

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define what to observe
# -------------------------------------------------------------------------------------------------------------------- #

writeGrad = True
writeY = False

boundaryPatches = ['lowerWall']
boundaryQuantities = ['dU']
boundaryQuantitiesIndices = [[1]]
boundaryDimensions = [9]
boundaryLimits = [[2.0, 10.0]]

obs = ClassObservable(writeGrad=writeGrad, writeY=writeY, boundaryPatches=boundaryPatches,
                      boundaryQuantities=boundaryQuantities, boundaryQuantitiesIndices=boundaryQuantitiesIndices,
                      boundaryLimits=boundaryLimits, boundaryDimensions=boundaryDimensions)

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
Re = 200.0
hFOAM = 0.01
dimSpace = 2

T = 0.1
h = 0.05

uMin = [-0.2]
uMax = [0.2]
nGridU = 2  # number of parts the grid is split into (--> uGrid = [-2, 0, 2])

dimZ = 50

Ttrain = 100.0  # Time for the simulation in the traing data generation
nLag = 100  # Lag time for EDMD
nMonomials = 3  # Max order of monomials for EDMD

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

of = ClassOpenFOAM(problem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, dimZ=dimZ, nGridU=nGridU)

# u = np.random.rand(1, int(T / m.h) + 1)
u = np.zeros([int(T / model.h + 1), 1], dtype=float)

[y, z, t, m] = model.integrate(None, u, 0.0)
[y2, z2, t2, m] = model.integrate(None, u, t[-1])

of.cleanCase()

plot(z={'t': t, 'z': z, 'iplot': 0}, z2={'t': t2, 'z2': z2, 'iplot': 0})

print('Done')
