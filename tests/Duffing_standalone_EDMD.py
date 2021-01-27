import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from scipy.optimize import minimize
from scipy.optimize import Bounds

# some stuff for nice plots
font = {'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #
h = 1e-2                        # Time step of integrator
facU = 2                        # Factor by which dt is larger
dt = facU * h                   # Coarse time step for SUR
T = 1.0                         # Final time
dimY = 2                        # dimension of state space

u_min = -4.0                    # lower bound for control
u_max = 4.0                     # upper bound for control

V = [u_min, u_max]              # finite set of admissible controls in (II) and (III)
nu = len(V)                     # dimension of V

Q = [0.0, 1.0]                  # weights on the diagonal of the Q-matrix in tje objective function

y0 = [1.0, 1.0]                 # initial condition for y

nt = round(T / h) + 1           # number of time steps on fine grid
nt2 = round(T / dt) + 1         # number of time steps on coarser grid for SUR

t = np.linspace(0.0, T, nt)     # array of time steps (fine grid)
t2 = np.linspace(0.0, T, nt2)   # array of time steps (coarse grid)

u0 = np.zeros(nt, dtype=float)  # initial guess for control u

y_ref = np.zeros((nt, 2), dtype=float)      # reference trajectory on fine grid
y_ref2 = y_ref[::facU, :]                   # reference trajectory on coarse grid


# -------------------------------------------------------------------------------------------------------------------- #
# Function that maps array from coarse to fine grid
# -------------------------------------------------------------------------------------------------------------------- #
def coarseGridToFine(x):
    if facU == 1:
        return x
    y = np.zeros(nt, dtype=float)
    for ii in range(nt2 - 1):
        y[facU * ii: facU * (ii + 1)] = x[ii]
    y[-1] = x[-1]
    return y


# -------------------------------------------------------------------------------------------------------------------- #
# ODE: Define right-hand side, ODE integrator and time-T-map Phi
# -------------------------------------------------------------------------------------------------------------------- #
def rhs(y_, u_):
    alpha, beta, delta = -1.0, 1.0, 0.0
    return np.array([y_[1], -delta * y_[1] - alpha * y_[0] - beta * y_[0] * y_[0] * y_[0] + u_])


def ODE(u_, y0_):
    y_ = np.zeros((u_.shape[0], dimY), dtype=float)
    y_[0, :] = y0_
    for ii in range(u_.shape[0] - 1):
        k1 = rhs(y_[ii, :], u_[ii])
        k2 = rhs(y_[ii, :] + h / 2 * k1[:], 0.5 * (u_[ii] + u_[ii]))
        k3 = rhs(y_[ii, :] + h / 2 * k2[:], 0.5 * (u_[ii] + u_[ii]))
        k4 = rhs(y_[ii, :] + h * k3[:], u_[ii])
        y_[ii + 1, :] = y_[ii, :] + h / 6 * (k1[:] + 2 * k2[:] + 2 * k3[:] + k4[:])
    return y_


def Phi(u_, y0_):
    # Integration with constant input over one time step of the coarse grid
    u2 = u_ * np.ones((facU + 1), dtype=float)
    y_ = ODE(u2, y0_)
    return y_[-1, :]


# -------------------------------------------------------------------------------------------------------------------- #
# Objective function for problem (I)
# -------------------------------------------------------------------------------------------------------------------- #
def J(u_):
    dy = ODE(u_, y0) - y_ref
    dyQ = np.zeros(dy.shape[0], dtype=float)
    for ii in range(dy.shape[1]):
        dyQ += Q[ii] * np.power(dy[:, ii], 2)

    return h * np.sum(dyQ)


# -------------------------------------------------------------------------------------------------------------------- #
# Data collection
# -------------------------------------------------------------------------------------------------------------------- #
y_min = [-2.0, -2.0]    # lower left corner for box of random initial conditions
y_max = [2.0, 2.0]      # upper right corner for box of random initial conditions
nData = 10000           # number of random initial conditions

# Create nData random initial conditions ...
X = np.ones((dimY, nData), dtype=float)
for i in range(dimY):
    X[i, :] *= y_min[i] + (y_max[i] - y_min[i]) * np.random.rand(nData)

# ... and evaluate Phi for all controls in V
Y = np.zeros((dimY, nData, nu), dtype=float)
for j in range(nu):
    for i in range(nData):
        Y[:, i, j] = Phi(V[j], X[:, i])

# -------------------------------------------------------------------------------------------------------------------- #
# Surrogate model via EDMD with polynomial dictionary
# -------------------------------------------------------------------------------------------------------------------- #
maxOrder = 3            # maximum order of polynomial terms


# lifts input matrix X to PsiX, containing polynomials of up to order maxOrder
# the array iy denotes the indices for projecting PsiX to X
# if maxOrder is set to zero, then PsiX = X, meaning that we use DMD
def psi(X_):
    if maxOrder == 0:
        iy_ = [0, 1]
        return X_, iy_
    else:
        PsiX_ = np.ones((100, X_.shape[1]), dtype=float)
        PsiX_[1:3, :] = X_

        s = 3
        for jj in range(2, maxOrder + 1):
            for ii in range(jj + 1):
                PsiX_[s, :] = np.power(X_[1, :], ii) * np.power(X_[0, :], jj - ii)
                s += 1

        iy_ = [1, 2]
        return PsiX_[:s, :], iy_


PsiX, iy = psi(X)       # Lift X to PsiX
G = PsiX @ PsiX.T       # define G-Matrix for EDMD

dimZ = PsiX.shape[0]    # dimension of the lifted state

# Compute Koopman matrices for the individual systems wit fixed controls u in V
K = np.zeros((dimZ, dimZ, nu), dtype=float)
for i in range(nu):
    PsiY, _ = psi(Y[:, :, i])
    Ai = PsiX @ PsiY.T
    K[:, :, i] = (pinv(G) @ Ai).T

# calculat lifted initial condition z0
y0a = np.zeros((2, 1), dtype=float)
y0a[:, 0] = y0
z0a, _ = psi(y0a)
z0 = z0a[:, 0]


# -------------------------------------------------------------------------------------------------------------------- #
# Integrator and Objective for surrogate model
# -------------------------------------------------------------------------------------------------------------------- #
def ROMAlpha(alpha, z0_):
    z_ = np.zeros((alpha.shape[0], dimZ), dtype=float)
    z_[0, :] = z0_

    # state at next time step is computed via a convex combination of the autonomous systems
    for ii in range(alpha.shape[0] - 1):
        z_[ii + 1, :] += (alpha[ii] * K[:, :, 0] + (1.0 - alpha[ii]) * K[:, :, 1]) @ z_[ii, :]
    return z_


def Jalpha(alpha_):
    z_ = ROMAlpha(alpha_, z0)
    dz = z_[:, iy] - y_ref2
    dzQ = np.zeros(dz.shape[0], dtype=float)
    for ii in range(dz.shape[1]):
        dzQ += Q[ii] * np.power(dz[:, ii], 2)

    return dt * np.sum(dzQ)


# -------------------------------------------------------------------------------------------------------------------- #
# Sum up rounding algorithm by Sager, Bock & Diehl
# -------------------------------------------------------------------------------------------------------------------- #
def SUR(alpha_):

    omega = np.zeros((nt2, 2), dtype=float)
    omegaHat = np.zeros(nu)

    for ii in range(nt2):
        for jj in range(nu - 1):
            omegaHat[jj] = np.sum(alpha_[:ii + 1]) - np.sum(omega[:ii, jj])
        omegaHat[-1] = np.sum(1.0 - alpha_[:ii + 1]) - np.sum(omega[:ii, -1])
        iOut = np.argmax(omegaHat)
        omega[ii, iOut] = 1.0

    u2 = np.zeros(nt2, dtype=float)
    for ii in range(nt2):
        for jj in range(nu):
            u2[ii] += omega[ii, jj] * V[jj]

    return u2


# -------------------------------------------------------------------------------------------------------------------- #
# Compare trajectories of full and surrogate model
# -------------------------------------------------------------------------------------------------------------------- #
alpha_test2 = np.random.rand(nt2)
u_test2 = V[0] * alpha_test2 + V[1] * (np.ones(nt2, dtype=float) - alpha_test2)
u_test = coarseGridToFine(u_test2)

y_test = ODE(u_test, y0)
z_test = ROMAlpha(alpha_test2, z0)

# -------------------------------------------------------------------------------------------------------------------- #
# Visualization of the comparison
# -------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure()
ax = list()
ax.append(fig.add_subplot(2, 2, 1))
ax.append(fig.add_subplot(2, 2, 2))
ax.append(fig.add_subplot(2, 2, 3))
ax.append(fig.add_subplot(2, 2, 4))
ax[0].plot(t, y_test[:, 0], linewidth=2, color='tab:blue', linestyle='solid', label=r'ODE')
ax[0].plot(t2, z_test[:, iy[0]], linewidth=2, color='tab:orange', linestyle='solid', label=r'Surrogate')
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$y_0$')
ax[0].grid()
ax[0].legend()
ax[1].plot(t, y_test[:, 1], linewidth=2, color='tab:blue', linestyle='solid', label=r'ODE')
ax[1].plot(t2, z_test[:, iy[1]], linewidth=2, color='tab:orange', linestyle='solid', label=r'Surrogate')
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$y_1$')
ax[1].grid()
ax[1].legend()
ax[2].plot(t, u_test, linewidth=2, color='tab:blue', linestyle='solid', label=r'ODE')
ax[2].plot(t2, u_test2, linewidth=2, color='tab:orange', linestyle='solid', label=r'Surrogate')
ax[2].set_xlabel(r'$t$')
ax[2].set_ylabel(r'$u$')
ax[2].grid()
ax[2].legend()
ax[3].plot(t2, np.absolute(y_test[::2, 0]-z_test[:, iy[0]]), linewidth=2, color='k', linestyle='solid', label=r'$\Delta_0$')
ax[3].plot(t2, np.absolute(y_test[::2, 1]-z_test[:, iy[1]]), linewidth=2, color='k', linestyle='dashed', label=r'$\Delta_1$')
ax[3].set_xlabel(r'$t$')
ax[3].set_ylabel(r'$|y - z|$')
ax[3].grid()
ax[3].legend()
fig.suptitle(r'Comparison of true model and surrogate model')
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Solve optimization problem (I)
# -------------------------------------------------------------------------------------------------------------------- #
# box constraints u_min <= u <= u_max
bounds = Bounds(u_min * np.ones(nt, dtype=float), u_max * np.ones(nt, dtype=float))

# call optimizer
res = minimize(J, u0, method='SLSQP', bounds=bounds)

# extract u, J and calculate y
JI_opt = res.fun
uI_opt = res.x
yI_opt = ODE(uI_opt, y0)

# -------------------------------------------------------------------------------------------------------------------- #
# Solve optimization problem (IV)
# -------------------------------------------------------------------------------------------------------------------- #
# initial value for alpha on coarse grid
alpha02 = 0.5 * np.ones(nt2, dtype=float)

# alpha is bounded between 0 and 1
bounds = Bounds(0.0 * np.ones(nt2, dtype=float), 1.0 * np.ones(nt2, dtype=float))

# call optimizer
res = minimize(Jalpha, alpha02, method='SLSQP', bounds=bounds)

# extract alpha and J
JIV_opt = res.fun
alpha_opt2 = res.x

# in the continuous case, u is obtaind by a convex combination of the entries in V and using alpha
uIV_opt2 = alpha_opt2 * V[0] + (1.0 - alpha_opt2) * V[1]
uIV_opt = coarseGridToFine(uIV_opt2)
yIV_opt = ODE(uIV_opt, y0)

# in the mixed-integer case, u is in V at all times, which is ensured by sum-up rounding
uIII_opt2 = SUR(alpha_opt2)
uIII_opt = coarseGridToFine(uIII_opt2)
yIII_opt = ODE(uIII_opt, y0)

# -------------------------------------------------------------------------------------------------------------------- #
# Visualization of the optimal trajectories
# -------------------------------------------------------------------------------------------------------------------- #
fig = plt.figure()
ax = list()
ax.append(fig.add_subplot(2, 2, 1))
ax.append(fig.add_subplot(2, 2, 2))
ax.append(fig.add_subplot(2, 2, 3))
ax.append(fig.add_subplot(2, 2, 4))
ax[0].plot(t, yI_opt[:, 0], linewidth=2, color='tab:blue', linestyle='solid', label=r'$y_0(I)$')
ax[0].plot(t, yIII_opt[:, 0], linewidth=2, color='tab:orange', linestyle='solid', label=r'$y_0(III)$')
ax[0].plot(t, yIV_opt[:, 0], linewidth=2, color='tab:green', linestyle='solid', label=r'$y_0(IV)$')
ax[0].plot(t, y_ref[:, 1], linewidth=2, color='k', linestyle='dashed', label=r'$y_{1,ref}$')
ax[0].plot(t, yI_opt[:, 1], linewidth=2, color='tab:blue', linestyle='dashed', label=r'$y_1(I)$')
ax[0].plot(t, yIII_opt[:, 1], linewidth=2, color='tab:orange', linestyle='dashed', label=r'$y_1(III)$')
ax[0].plot(t, yIV_opt[:, 1], linewidth=2, color='tab:green', linestyle='dashed', label=r'$y_1(IV)$')
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$y_0$')
ax[0].grid()
ax[0].legend(fontsize=12)
ax[1].plot(t, uI_opt, linewidth=2, color='tab:blue', linestyle='solid', label=r'(I)')
ax[1].plot(t, uIII_opt, linewidth=2, color='tab:orange', linestyle='solid', label=r'(III)')
ax[1].plot(t, uIV_opt, linewidth=2, color='tab:green', linestyle='solid', label=r'(IV)')
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$u$')
ax[1].grid()
ax[1].legend()
ax[2].plot(t, np.sqrt(np.sum(np.power(yI_opt - yIII_opt, 2), axis=1)), linewidth=2, color='tab:orange', linestyle='solid', label=r'(III)')
ax[2].plot(t, np.sqrt(np.sum(np.power(yI_opt - yIV_opt, 2), axis=1)), linewidth=2, color='tab:green', linestyle='solid', label=r'(IV)')
ax[2].set_xlabel(r'$t$')
ax[2].set_ylabel(r'$\|y - y(I)\|_2$')
ax[2].grid()
ax[2].legend()
ax[3].plot(t, np.absolute(uIII_opt - uI_opt), linewidth=2, color='tab:orange', linestyle='solid', label=r'(III)')
ax[3].plot(t, np.absolute(uIV_opt - uI_opt), linewidth=2, color='tab:green', linestyle='solid', label=r'(IV)')
ax[3].set_xlabel(r't')
ax[3].set_ylabel(r'$|u - u(I)|$')
ax[3].grid()
ax[3].legend()
fig.suptitle(r'Comparison different optimal solutions')
plt.show()
