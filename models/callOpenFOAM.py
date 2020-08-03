import os
from numpy import concatenate


def simulateModel(y0, t0, u, model):
    nt = u.shape[0]
    T = t0 + nt * model.h

    print('Solving model via OpenFOAM from t = {:.2f} to t = {:.2f}'.format(t0, T))

    if t0 < 1e-3:
        model.OF.cleanCase()

    model.OF.writeControlFile(u, T, t0)
    model.OF.writeTransportPropertiesFile()
    model.OF.writeControlDictFile(T, t0, model.h)

    os.system('bash -c "export LD_LIBRARY_PATH={}; {}runFoam {} {} > {}/log"'.format(model.OF.pathLib,
                                                                                     model.OF.pathOFOut,
                                                                                     model.OF.pathOFOut,
                                                                                     model.OF.nProc,
                                                                                     model.OF.pathOFOut))

    z = list()
    if model.OF.obs.flagForceCoeffs:
        [z1, t1] = model.OF.readForces(t0, T, model.h)
        z.append(z1)
    if model.OF.obs.flagProbes:
        [z1, t1] = model.OF.readProbes(t0, T, model.h)
        z.append(z1)
    if model.OF.obs.flagBoundaries:
        if model.OF.nProc > 1:
            os.system(
                'bash -c "export LD_LIBRARY_PATH={}; {}runReconstructPar {} {} {} {}"'.format(
                    model.OF.pathLib, model.OF.pathOFOut, model.OF.pathOFOut, t0, T, model.OF.pathOFOut))
        [z1, t1] = model.OF.readBoundaries(t0, T, model.h)
        z.append(z1)
    if model.OF.obs.flagFullState:
        [z1, t1] = model.OF.readSolution(t0, T, model.h)
        z.append(z1)
    z = concatenate(z, axis=1)

    if model.OF.obs.writeY:
        y, t = model.OF.readSolution(t0, T, model.h)
    else:
        y = []
        t = t1

    return y, z, t, model
