from QuaSiModO import *
from OpenFOAM import configOpenFOAM
import os
import shutil


class ClassOFMesh:
    """ClassOFMesh

    This class contains the mesh information
    - number of grid poitns
    - cell centers
    - cell volumes

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 04/2020
    """

    nc = []
    C = []
    V = []
    nSkipField = None
    flagSkipBoundariesSet = False


class ClassOpenFOAM:
    """
    ClassOpenFOAM

    This class contains the functionality to set up and solve OpenFOAM simulations both in serial and parallel mode.

    Input
    Mandatory:
    1) pathProblem: Path to the problem template
    2) obs: Observable object defining which data to extract from the simulation
    3) pathOut: Path into which the template is copied

    Optional:
    4) nProc(def = 1): number of processors
    5) nInputs(def = 1): number of inputs
    6) dimInputs(def = 1): dimension of the input(1 for scalar(e.g. rotation) or 3 for vector(e.g.inflow velocity))
    7) iInputs(def = 0): indices of how the inputs are distributed to the correct dimensions of the input in OF
    8) h (def = 0.01): Time step of OpenFOAM solver
    9) Re (def = 100): Reynolds number for simulation
    10) dimSpace (def = 2): Dimension of the spatial domain

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 04 / 2020
    """

    # Fixed paths to OpenFOAM
    pathOF, pathThirdParty = configOpenFOAM.getPaths()
    platform, platformThirdParty = configOpenFOAM.getPlatform()
    skipRowsForceCoeffs = configOpenFOAM.getSkipRows()

    # Numerical setup% % Create model file
    maxCo = configOpenFOAM.getCourant()
    solver = configOpenFOAM.getSolver()
    purgeWrite = 0

    mesh = ClassOFMesh()

    def __init__(self, pathProblem, obs, pathOut, nProc=1, nInputs=1, dimInputs=1, iInputs=[0], h=0.01, Re=100.0,
                 dimSpace=2, BCWrite='0'):

        self.obs = obs

        self.pathOut = pathOut
        if not self.pathOut[-1] == '/':
            self.pathOut = self.pathOut + '/'

        self.nProc = nProc
        if self.nProc == 1:
            self.pathLib = self.pathOF + 'platforms/' + self.platform + '/lib:' + \
                           self.pathOF + 'platforms/' + self.platform + '/lib/dummy'
        else:
            self.pathLib = self.pathOF + 'platforms/' + self.platform + '/lib:' + \
                           self.pathOF + 'platforms/' + self.platform + '/lib/openmpi-system'

        self.pathProblem = pathProblem
        self.modelFile = 'callOpenFOAM.py'

        self.nInputs = nInputs
        self.dimInputs = dimInputs
        self.iInputs = iInputs
        self.dimSpace = dimSpace
        self.BCWrite = BCWrite

        if h is not None:
            self.h = h

        if Re is not None:
            self.Re = Re

        # Create pathOut if not existent
        if not os.path.exists(self.pathOut):
            # os.mkdir(self.pathOut)
            os.makedirs(self.pathOut)

        self.pathOFOut = self.pathOut + 'FOAM/'

        if os.path.exists(self.pathOFOut):
            print('Warning: Folder {} exists, deleting'.format(self.pathOFOut))
            os.system('rm -rf {}'.format(self.pathOFOut))

        # Create Temp case
        shutil.copytree(self.pathProblem, self.pathOFOut)

        self.writeControlDictFile(1.0, 0.0, self.h)
        self.writeRunFoam()

        if self.obs.flagForceCoeffs:
            self.writeForceCoeffs()

        if self.obs.flagProbes:
            self.writeProbes()

        self.getMeshInfo()

        if self.nProc > 1:
            if obs.writeY or obs.flagFullState:
                self.writeReoncstructPar('U,dU')
            else:
                self.writeReoncstructPar('dU')
            if not os.path.exists(self.pathOFOut + 'processor0'):
                u = np.zeros([2, nInputs], dtype=float)
                self.writeControlFile(u, 1.0, 0.0)
                self.writeDecomposeParDictFile()
                self.writeDecomposePar()

                pathLibraries = self.pathOF + 'platforms/' + self.platform + '/lib:' + \
                                self.pathOF + 'platforms/' + self.platform + '/lib/dummy:' + \
                                self.pathThirdParty + 'platforms/' + self.platformThirdParty + '/lib'

                os.system('bash -c "export LD_LIBRARY_PATH=' + pathLibraries + '; ' + self.pathOFOut + 'runDP ' +
                          self.pathOFOut + ' > ' + self.pathOFOut + 'logDP"')

    def createOrUpdateModel(self, uMin, uMax, hWrite, dimZ=None, Re=None, iObs=None, y0=None, typeUGrid=None, nGridU=1,
                            uGrid=None):
        model = ClassModel(self, uMin, uMax, hWrite, dimZ, Re, iObs, y0, typeUGrid, nGridU, uGrid, self.obs.writeY)
        return model

    def createIC(self, y0):
        pass
        # writeVelocity(tempdir, Yp0)
        # writePressure(tempdir, Yp0)

    def writeField(self, Y, name, iStart=200):

        if len(Y.shape) == 1:
            Y = np.array([Y])
        dimY = int(round(Y.shape[1] / self.mesh.nc))

        linesP = list()
        flagWrite = True
        i = 0
        if dimY == 1:
            sampleFile = 'p'
        else:
            sampleFile = 'U'

        with open(self.pathOFOut + self.BCWrite + '/' + sampleFile) as fileObject:
            for line in fileObject:
                i += 1
                if line.find('boundaryField') >= 0:
                    flagWrite = True
                if flagWrite:
                    linesP.append(line)
                if line.find('internalField') >= 0:
                    flagWrite = False
                    iPartOne = i

        nt = Y.shape[0]
        indices = np.asarray(range(0, Y.shape[1], dimY))
        ny = len(indices)
        filesOut = list()
        for i in range(nt):
            filesOut.append(list())
            os.makedirs(self.pathOFOut + str(iStart + i))

            fout = open(self.pathOFOut + str(iStart + i) + '/' + name, 'w')
            for k in range(iPartOne):
                fout.write(linesP[k])

            fout.write('{}\n'.format(ny))

            fout.write('(\n')
            if dimY == 1:
                for k in range(ny):
                    fout.write('{}\n'.format(Y[i, k]))
            elif dimY == 2:
                for k in range(ny):
                    fout.write('({} {} 0)\n'.format(Y[i, k * dimY], Y[i, k * dimY + 1]))
            else:
                for k in range(ny):
                    fout.write('({} {} {}})\n'.format(Y[i, k * dimY], Y[i, k * dimY + 1], Y[i, k * dimY + 2]))

            fout.write(')\n;\n\n')
            for k in range(iPartOne, len(linesP)):
                fout.write(linesP[k])
            fout.close()

    def calcGradient(self, Y):

        linesP = list()
        flagWrite = True
        i = 0
        with open(self.pathOFOut + self.BCWrite + '/p') as fileObject:
            for line in fileObject:
                i += 1
                if line.find('boundaryField') >= 0:
                    flagWrite = True
                if flagWrite:
                    linesP.append(line)
                if line.find('internalField') >= 0:
                    flagWrite = False
                    iPartOne = i

        dimY = int(round(Y.shape[1] / self.mesh.nc))
        nt = Y.shape[0]
        indices = np.asarray(range(0, Y.shape[1], dimY))
        ny = len(indices)
        filesOut = list()
        for i in range(nt):
            filesOut.append(list())
            os.makedirs(self.pathOFOut + str(100 + i))
            for j in range(dimY):
                fout = open(self.pathOFOut + str(100 + i) + '/Y' + str(j), 'w')
                for k in range(iPartOne):
                    fout.write(linesP[k])

                fout.write('{}\n'.format(ny))
                fout.write('(\n')
                for k in range(ny):
                    fout.write('{}\n'.format(Y[i, indices[k] + j]))
                fout.write(')\n;\n\n')
                for k in range(iPartOne, len(linesP)):
                    fout.write(linesP[k])
                fout.close()

        self.writeRunGrad(dimY, 99 + nt)
        currentDir = os.getcwd()
        os.chdir(self.pathOFOut)
        os.system('bash -c "export LD_LIBRARY_PATH={}; ./runGrad > logGrad"'.format(self.pathLib))
        os.chdir(currentDir)

        dY = np.zeros([nt, ny, dimY, self.dimSpace], dtype=float)
        for j in range(dimY):
            dYj, _ = self.readSolutionTensor(100, 99 + nt, 1, whichField='dY{}'.format(j))
            for k in range(self.dimSpace):
                dY[:, :, j, k] = dYj[:, :, k]

        for i in range(nt):
            os.system('rm -rf {}'.format(self.pathOFOut + str(100 + i)))
        os.system('rm -rf {}runGrad {}dictGrad {}logGrad'.format(self.pathOFOut, self.pathOFOut, self.pathOFOut))

        return dY

    def writeRunGrad(self, dimY, T):

        fout = open(self.pathOFOut + 'runGrad', 'w')
        fout.write('source ' + self.pathOF + 'etc/bashrc\n')
        strOut = 'postProcess -dict dictGrad -fields "('
        for i in range(dimY):
            strOut += 'Y{} '.format(i)
        strOut = strOut[:-1] + ')" -time "100:{}"\n'.format(T)
        fout.write(strOut)
        fout.close()

        os.system('chmod 777 {}'.format(self.pathOFOut + 'runGrad'))

        fout = open(self.pathOFOut + 'dictGrad', 'w')
        fout.write('FoamFile\n')
        fout.write('{\n')
        fout.write('    version     2.0;\n')
        fout.write('    format      ascii;\n')
        fout.write('    class       dictionary;\n')
        fout.write('    location    $FOAM_CASE;\n')
        fout.write('    object      controlDict;\n')
        fout.write('}\n')
        fout.write('application     {};\n'.format(self.solver))
        fout.write('startFrom       startTime;\n')
        fout.write('startTime       100;\n')
        fout.write('stopAt          endTime;\n')
        fout.write('endTime         {};\n'.format(T))
        fout.write('deltaT          1;\n')
        fout.write('maxDeltaT	   1;\n')
        fout.write('adjustTimeStep  no;\n')
        fout.write('writeInterval   1;\n')
        fout.write('writeFormat     ascii;\n')
        fout.write('writePrecision  14;\n')
        fout.write('writeCompression off;\n')
        fout.write('timeFormat      general;\n')
        fout.write('timePrecision   12;\n')
        fout.write('functions\n')
        fout.write('{\n')
        for i in range(dimY):
            fout.write('    grad{}\n'.format(i))
            fout.write('    {\n')
            fout.write('        type grad;\n')
            fout.write('        libs           ("libfieldFunctionObjects.so");\n')
            fout.write('        field          "Y{}";\n'.format(i))
            fout.write('        result         "dY{}";\n'.format(i))
            fout.write('        executeControl timeStep;\n')
            fout.write('        writeControl   writeTime;\n')
            fout.write('    }\n')
        fout.write('}\n')
        fout.close()

    def writeRunFoam(self):

        fout = open(self.pathOFOut + 'runFoam', 'w')
        fout.write('source ' + self.pathOF + 'etc/bashrc\n')
        fout.write('if [ $# -lt 2 ]\n')
        fout.write('then \n')
        fout.write('    np=1\n')
        fout.write('else\n')
        fout.write('    np=$2\n')
        fout.write('fi\n')
        fout.write('if [ $np -eq 1 ]\n')
        fout.write('then\n')
        fout.write('    ' + self.solver + ' -case $1\n')
        fout.write('else\n')
        fout.write('    mpirun -np $np ' + self.solver + ' -case $1 -parallel\n')
        fout.write('fi\n')
        fout.close()

        os.system('chmod 777 {}'.format(self.pathOFOut + 'runFoam'))

    def writeRunWCC(self):

        fout = open(self.pathOFOut + 'runWCC', 'w')
        fout.write('source ' + self.pathOF + 'etc/bashrc\n')
        fout.write('postProcess -func writeCellCentres -case $1\n')
        fout.write('postProcess -func writeCellVolumes -case $1\n')
        fout.close()

        os.system('chmod 777 {}'.format(self.pathOFOut + 'runWCC'))

    def writeDecomposePar(self):

        fout = open(self.pathOFOut + 'runDP', 'w')
        fout.write('source ' + self.pathOF + 'etc/bashrc\n')
        fout.write('decomposePar -case $1\n')
        fout.close()

        os.system('chmod 777 {}'.format(self.pathOFOut + 'runDP'))

    def writeControlFile(self, u, T, t0):
        # Writes a control file with name "control{i}" for i = 1 : nInputs. For every control file, the given number of
        # inputs is written to the indices given in iInput.All other entries are set to zero.
        nt = u.shape[0]
        t = np.linspace(t0, T, nt)

        u2 = np.zeros([2 * u.shape[0] - 1, u.shape[1]], dtype=float)
        t2 = np.zeros([u2.shape[0]])

        for ii in range(u.shape[0] - 1):
            t2[2 * ii] = t[ii]
            t2[2 * ii + 1] = t[ii + 1] - 1e-5
            u2[2 * ii, :] = u[ii, :]
            u2[2 * ii + 1, :] = u[ii, :]

        u2[-1, :] = u[-1, :]
        t2[-1] = t[-1]

        dimGiven = int(u.shape[1] / self.nInputs)
        for j in range(self.nInputs):
            fout = open(self.pathOFOut + 'control' + str(j), 'w')
            fout.write('(\n')

            u3 = np.zeros([u2.shape[0], self.dimInputs])
            u3[:, self.iInputs] = u2[:, j * dimGiven: (j + 1) * dimGiven]
            if self.dimInputs > 1:
                for ii in range(t2.shape[0]):
                    fout.write('({} ({} {} {}))\n'.format(t2[ii], u3[ii, 0], u3[ii, 1], u3[ii, 2]))
            else:
                for ii in range(t2.shape[0]):
                    fout.write('({} {})\n'.format(t2[ii], u3[ii, 0]))
            fout.write(')\n')
            fout.close()

    def writeDecomposeParDictFile(self):

        fout = open(self.pathOFOut + 'system/decomposeParDict', 'w')
        fout.write('FoamFile\n')
        fout.write('{\n')
        fout.write('    version     2.0;\n')
        fout.write('    format      ascii;\n')
        fout.write('    class       dictionary;\n')
        fout.write('    location    "system";\n')
        fout.write('    object      decomposeParDict;\n')
        fout.write('}\n')
        fout.write('numberOfSubdomains {};\n'.format(self.nProc))
        fout.write('method          scotch;\n')
        # fout.write('scotchCoeffs\n')
        # fout.write('{\n')
        # fout.write('}\n')
        # fout.write('distributed     no;\n')
        # fout.write('roots           ( )\n')
        fout.close()

    def writeReoncstructPar(self, fields='dU'):

        fout = open(self.pathOFOut + 'runReconstructPar', 'w')
        fout.write('source ' + self.pathOF + 'etc/bashrc\n')
        fout.write('reconstructPar -case $1 -time $2:$3 -fields \'(' + fields + ')\' > $4/logRP\n')
        fout.close()

        os.system('chmod 777 {}'.format(self.pathOFOut + 'runReconstructPar'))

    def writeControlDictFile(self, T, t0, hWrite):

        filename = self.pathOFOut + 'system/controlDict'

        fout = open(filename, 'w')
        fout.write('FoamFile\n')
        fout.write('{\n')
        fout.write('    version     2.0;\n')
        fout.write('    format      ascii;\n')
        fout.write('    class       dictionary;\n')
        fout.write('    location    "system";\n')
        fout.write('    object      controlDict;\n')
        fout.write('}\n')
        fout.write('application     {};\n'.format(self.solver))
        fout.write('startFrom       startTime;\n')
        fout.write('startTime       {};\n'.format(t0))
        fout.write('stopAt          endTime;\n')

        if self.maxCo > 2:
            fout.write('endTime         {};\n'.format(T))
        else:
            fout.write('endTime         {};\n'.format(T + self.h))

        fout.write('deltaT          {};\n'.format(self.h))
        fout.write('maxDeltaT	   {};\n'.format(hWrite))

        if self.maxCo > 2:
            fout.write('adjustTimeStep  no;\n')
        else:
            fout.write('adjustTimeStep  yes;\n')

        if self.purgeWrite > 0:
            fout.write('purgeWrite      {};\n'.format(self.purgeWrite))

        fout.write('maxCo           {};\n'.format(self.maxCo))
        fout.write('writeControl    adjustableRunTime;\n')
        fout.write('writeInterval   {};\n'.format(hWrite))
        fout.write('writeFormat     ascii;\n')
        fout.write('writePrecision  14;\n')
        fout.write('writeCompression off;\n')
        fout.write('timeFormat      general;\n')
        fout.write('timePrecision   12;\n')
        fout.write('runTimeModifiable true;\n')
        fout.write('functions\n')
        fout.write('{\n')

        if self.obs.flagForceCoeffs:
            fout.write('    #include "forceCoeffs"\n')

        if self.obs.flagProbes:
            fout.write('    #include "probes"\n')

        if self.obs.writeGrad:
            fout.write('    gradU\n')
            fout.write('    {\n')
            fout.write('        type grad;\n')
            fout.write('        libs ("libfieldFunctionObjects.so");\n')
            fout.write('        field     "U";\n')
            fout.write('        result    "dU";\n')
            fout.write('        executeControl  timeStep;\n')
            fout.write('        writeControl    writeTime;\n')
            fout.write('    }\n')

        fout.write('}\n')
        fout.close()

    def writeTransportPropertiesFile(self):

        filename = self.pathOFOut + 'constant/transportProperties'
        fout = open(filename, 'w')
        fout.write('FoamFile\n')
        fout.write('{\n')
        fout.write('    version     2.0;\n')
        fout.write('    format      ascii;\n')
        fout.write('    class       dictionary;\n')
        fout.write('    location    "constant";\n')
        fout.write('    object      transportProperties;\n')
        fout.write('}\n')
        fout.write('transportModel  Newtonian;\n')
        fout.write('nu              nu [ 0 2 -1 0 0 0 0 ] {};\n'.format(1.0 / self.Re))
        fout.write('CrossPowerLawCoeffs\n')
        fout.write('{\n')
        fout.write('    nu0             nu0 [ 0 2 -1 0 0 0 0 ] 1e-06;\n')
        fout.write('    nuInf           nuInf [ 0 2 -1 0 0 0 0 ] 1e-06;\n')
        fout.write('    m               m [ 0 0 1 0 0 0 0 ] 1;\n')
        fout.write('    n               n [ 0 0 0 0 0 0 0 ] 1;\n')
        fout.write('}\n')
        fout.write('BirdCarreauCoeffs\n')
        fout.write('{\n')
        fout.write('    nu0             nu0 [ 0 2 -1 0 0 0 0 ] 1e-06;\n')
        fout.write('    nuInf           nuInf [ 0 2 -1 0 0 0 0 ] 1e-06;\n')
        fout.write('    k               k [ 0 0 1 0 0 0 0 ] 0;\n')
        fout.write('    n               n [ 0 0 0 0 0 0 0 ] 1;\n')
        fout.write('}\n')
        fout.close()

    def writeForceCoeffs(self):

        filename = self.pathOFOut + 'system/forceCoeffs'
        fout = open(filename, 'w')
        for i in range(len(self.obs.forceCoeffsPatches)):
            fout.write('forceCoeffs{}\n'.format(i))
            fout.write('{\n')
            fout.write('type forceCoeffs;\n')
            fout.write('libs ("libforces.so");\n')
            fout.write('writeControl timeStep;\n')
            fout.write('timeInterval 1;\n')
            fout.write('log yes;\n')
            fout.write('patches (\n')
            fout.write('        {}\n'.format(self.obs.forceCoeffsPatches[i]))
            fout.write('       );\n')

            fout.write('rho rhoInf;\n')
            fout.write('rhoInf 1;\n')
            fout.write('liftDir     (0 1 0);\n')
            fout.write('dragDir     (1 0 0);\n')
            fout.write('origin      (0 0 0);\n')
            fout.write('CofR        (0 0 0);\n')
            fout.write('pitchAxis   (0 0 0);\n')
            fout.write('magUInf     1;\n')
            fout.write('lRef        {};\n'.format(self.obs.lRef))
            fout.write('Aref        {};\n'.format(self.obs.ARef))
            fout.write('}\n')
        fout.close()

    def writeProbes(self):

        filename = self.pathOFOut + 'system/probes'
        fout = open(filename, 'w')
        fout.write('probes\n')
        fout.write('{\n')
        fout.write('libs ( "libsampling.so" );\n')
        fout.write('type        probes;\n')
        fout.write('name        probes;\n')
        fout.write('fields (\n')
        for i in range(len(self.obs.probeQuantities)):
            fout.write('        {}\n'.format(self.obs.probeQuantities[i]))
        fout.write('       );\n')
        fout.write('\n')
        fout.write('probeLocations\n')
        fout.write('(\n')
        for i in range(len(self.obs.probeLocations)):
            fout.write('({} {} {})\n'.format(self.obs.probeLocations[i, 0], self.obs.probeLocations[i, 1],
                                             self.obs.probeLocations[i, 2]))
        fout.write(');\n')
        fout.write('}\n')
        fout.close()

    def readForces(self, t0, T, h):
        nt = int(round((T - t0) / h)) + 1
        tout = np.linspace(t0, T, nt)
        zout = np.zeros([len(tout), 2 * len(self.obs.forceCoeffsPatches)], dtype=float)
        for i in range(len(self.obs.forceCoeffsPatches)):
            for fileName in os.listdir(self.pathOFOut + 'postProcessing/forceCoeffs{}'.format(i)):
                if abs(float(fileName) - t0) < 1e-4:
                    fin = open(self.pathOFOut + 'postProcessing/forceCoeffs{}/{}/coefficient.dat'.format(i, fileName), 'r')
                    A = np.loadtxt(fin, skiprows=self.skipRowsForceCoeffs)
                    t = A[:, 0]
                    z = A[:, [1, 3]]
                    fin.close()

                    f = interpolate.interp1d(t, z, axis=0, kind='linear', copy=True, bounds_error=None,
                                             fill_value='extrapolate')
                    zout[:, 2 * i: 2 * (i + 1)] = f(tout)

        return zout, tout

    def readProbes(self, t0, T, h):
        z = list()
        t = list()
        nt = np.zeros([len(self.obs.probeQuantities)], dtype=int)
        tout = np.linspace(t0, T, int(round((T - t0) / h)) + 1)
        for i in range(len(self.obs.probeQuantities)):
            for fileName in os.listdir(self.pathOFOut + 'postProcessing/probes'):
                try:
                    tt = float(fileName)
                    if tt < t0 - 1e-6:
                        continue
                except Exception:
                    continue
                fileStr = self.pathOFOut + 'postProcessing/probes/{}/{}'.format(fileName, self.obs.probeQuantities[i])
                if os.path.exists(fileStr):
                    fin = open(fileStr, 'r')
                    A = np.loadtxt(reformatLines(fin), skiprows=2 + self.obs.probeLocations.shape[0])
                    t.append(A[:, 0])
                    nt[i] = len(t[i])

                    nP = len(self.obs.probeLocations)
                    indices = []
                    for j in range(nP):
                        indices.append(np.array(self.obs.probeQuantitiesIndices[i]))
                        for k in range(len(indices[-1])):
                            indices[-1][k] += int(1.0 + float(j) * float(self.obs.probeDimensions[i]))
                    z.append(A[:, np.concatenate(indices, axis=0)])
                    fin.close()

        nMin = min(nt)
        for i in range(len(nt)):
            if len(t[i]) > nMin:
                z[i] = z[i][len(t[i]) - nMin:, :]
                t[i] = t[i][len(t[i]) - nMin:]
        z = np.concatenate(z, axis=1)
        f = interpolate.interp1d(t[0], z, axis=0, kind='linear', copy=True, bounds_error=None, fill_value='extrapolate')
        zout = f(tout)
        return zout, tout

    def readSolution(self, t0, T, h, whichField='U', dimY=None):

        if dimY is None:
            dimY = self.dimSpace

        indices = np.asarray(range(0, dimY * self.mesh.nc, dimY))
        nt = int(round((T - t0) / h)) + 1
        t = np.linspace(t0, T, nt)
        Y = np.zeros([nt, dimY * self.mesh.nc], dtype=float)

        for fileName in os.listdir(self.pathOFOut):
            try:
                tt = float(fileName)

                # min(myList, key=lambda x: abs(x - myNumber))
                j = min(range(len(t)), key=lambda i: abs(t[i] - tt))
                minVal = abs(t[j] - tt)

                if minVal < 1e-5:
                    if self.mesh.nSkipField is None:
                        fin = open(self.pathOFOut + fileName + '/{}'.format(whichField), 'r')
                        nSkip = 0
                        while True:
                            line = fin.readline()
                            nSkip += 1
                            if line.find('internalField') >= 0:
                                self.mesh.nSkipField = nSkip + 2
                                fin.close()
                                break
                    fin = open(self.pathOFOut + fileName + '/{}'.format(whichField), 'r')
                    A = np.loadtxt(reformatLines(fin), skiprows=self.mesh.nSkipField, max_rows=self.mesh.nc)
                    fin.close()

                    for k in range(dimY):
                        Y[j, indices + k] = A[:, k]
            except Exception:
                pass

        return Y, t

    def readSolutionTensor(self, t0, T, h, whichField):

        nt = int(round((T - t0) / h)) + 1
        t = np.linspace(t0, T, nt)
        Y = np.zeros([nt, self.mesh.nc, self.dimSpace], dtype=float)

        for fileName in os.listdir(self.pathOFOut):
            try:
                tt = float(fileName)

                # min(myList, key=lambda x: abs(x - myNumber))
                j = min(range(len(t)), key=lambda i: abs(t[i] - tt))
                minVal = t[j] - tt

                if minVal < 1e-5:
                    if self.mesh.nSkipField is None:
                        fin = open(self.pathOFOut + fileName + '/{}'.format(whichField), 'r')
                        nSkip = 0
                        while True:
                            line = fin.readline()
                            nSkip += 1
                            if line.find('internalField') >= 0:
                                self.mesh.nSkipField = nSkip + 2
                                fin.close()
                                break
                    fin = open(self.pathOFOut + fileName + '/{}'.format(whichField), 'r')
                    A = np.loadtxt(reformatLines(fin), skiprows=self.mesh.nSkipField, max_rows=self.mesh.nc)
                    fin.close()
                    Y[j, :, :] = A[:, :self.dimSpace]
            except Exception:
                pass

        return Y, t

    def readBoundaries(self, t0, T, h):

        nt = int(round((T - t0) / h)) + 1
        t = np.linspace(t0, T, nt)

        flagZCreated = False

        for fileName in os.listdir(self.pathOFOut):
            try:
                tt = float(fileName)

                # min(myList, key=lambda x: abs(x - myNumber))
                k = min(range(len(t)), key=lambda i: abs(t[i] - tt))
                minVal = t[k] - tt

                if minVal < 1e-5:
                    for i in range(len(self.obs.boundaryQuantities)):
                        fileNameI = '{}{}/{}'.format(self.pathOFOut, fileName, self.obs.boundaryQuantities[i])
                        if not self.obs.flagNSet[i]:
                            for j in range(len(self.obs.boundaryPatches)):
                                self.obs.setN(fileNameI, i, j)
                            self.obs.flagNSet[i] = True

                        zt = []
                        for j in range(len(self.obs.boundaryPatches)):
                            fin = open(fileNameI, 'r')
                            A = np.loadtxt(reformatLines(fin), skiprows=self.obs.nSkip[j][i], max_rows=self.obs.nVar[j])
                            fin.close()
                            zt.append(A[self.obs.iVar[j], self.obs.boundaryQuantitiesIndices[i]])

                        if not flagZCreated:
                            zAll = np.concatenate(zt, axis=0)
                            z = np.zeros([nt, zAll.shape[0]])
                            z[k, :] = zAll
                            t[k] = tt
                            flagZCreated = True
                        else:
                            z[k, :] = np.concatenate(zt, axis=0)
                            t[k] = tt

            except Exception:
                pass

        return z, t

    def getMeshInfo(self):

        pathLibraries = self.pathOF + 'platforms/' + self.platform + '/lib:' + \
                        self.pathOF + 'platforms/' + self.platform + '/lib/dummy'
        self.writeRunWCC()
        os.system('bash -c "export LD_LIBRARY_PATH=' + pathLibraries + '; ' + self.pathOFOut + 'runWCC ' +
                  self.pathOFOut + ' > ' + self.pathOFOut + 'logWCC"')

        # Cell centers
        fin = open(self.pathOFOut + '0/C', 'r')
        self.mesh.nc = int(np.loadtxt(fin, skiprows=21, max_rows=1))
        self.mesh.C = np.loadtxt(reformatLines(fin), delimiter=' ', skiprows=1, max_rows=self.mesh.nc)
        fin.close()

        # Cell volumes
        self.mesh.V = np.loadtxt(self.pathOFOut + '0/V', delimiter=' ', skiprows=23, max_rows=self.mesh.nc)
        fin.close()

        # Remove ASCII files
        os.system('rm -f {}0/C {}0/Cx {}0/Cy {}0/Cz {}0/V'.format(self.pathOFOut, self.pathOFOut, self.pathOFOut,
                                                                  self.pathOFOut, self.pathOFOut))

    def cleanCase(self):
        os.system('rm -rf {}0.* {}postProcessing {}log*'.format(self.pathOFOut, self.pathOFOut, self.pathOFOut))
        for i in range(1, 9):
            os.system('rm -rf {}{}*'.format(self.pathOFOut, i))

        if self.nProc > 1:
            for j in range(self.nProc):
                os.system('rm -rf {}processor{}/0.* {}postProcessing {}log*'.format(self.pathOFOut, j, self.pathOFOut,
                                                                                    self.pathOFOut))
                for i in range(1, 9):
                    os.system('rm -rf {}processor{}/{}*'.format(self.pathOFOut, j, i))


class ClassObservable:
    """
    ClassObservable

    This class contains the functionality to obtain observables from the OpenFOAM solution

    Flags for the calculation of additional quantities:
    1) writeGrad (default = False): Compute the gradient 'dU' of the velocity field 'U'
    2) writeY (default = False): Read the entire velocity field U into the full state output y

    Possible types of observables:
    1) Boundary values, requires the following inputs
       - boundaryPatches = ['lowerWall']: Array of patch names
       - boundaryLimits = [[2.0, 10.0]]: 1D parametrization of the respective patches
       - boundaryLimitsReduced = [[3.0, 6.0]]: Limitation of the quantities to read to a subpart of the patch
       - boundaryQuantities = ['dU']: Array of Quantities to read on each patch
       - boundaryDimensions = [9]: Dimension of the respective quantities
       - boundaryQuantitiesIndices = [[1]]: Which indices to read from these quantities
    2) Lift and drag coefficient over a patch
       - forceCoeffsPatches = ['cylinder']: Array of patches from which to extract Cl and Cd
       - ARef = np.pi: Reference area
       - lRef = 1.0: Reference length
    3) Read data from probes at specified locations
       - probeLocations = [[5.0, 0.5, 0.5], [7.0, 0.5, 0.5]]: Array of 3D points to place probes
       - probeQuantities = ['U', 'dU']: Array of quantities to read at probe locations
       - probeDimensions = [3, 9]: Dimension of the respective quantities
       - probeQuantitiesIndices = [[0, 1], [0, 1, 3, 4]]: Indices of the respective quantites that are read
    4) Full state
       - flagFullState=true: writes the entire state of the PDE to the observable z

    Author: Sebastian Peitz, Paderborn University, speitz@math.upb.de

    First created: 05 / 2020
    """

    nSkip = []
    nVar = []
    iVar = []
    xVar = []
    flagNSet = []

    def __init__(self, probeQuantities=None, probeLocations=None, probeQuantitiesIndices=None, writeY=False,
                 probeDimensions=None, forceCoeffsPatches=None, boundaryPatches=None, boundaryQuantities=None,
                 boundaryLimits=None, boundaryLimitsReduced=None, writeGrad=False, boundaryQuantitiesIndices=None,
                 boundaryDimensions=None, lRef=1.0, ARef=np.pi, flagFullState=False):

        self.writeGrad = writeGrad
        self.writeY = writeY

        self.flagFullState = flagFullState

        if probeQuantities is not None:
            self.flagProbes = True
            self.probeQuantities = probeQuantities
            self.probeLocations = np.array(probeLocations)
            self.probeQuantitiesIndices = np.array(probeQuantitiesIndices)
            self.probeDimensions = np.array(probeDimensions)
        else:
            self.flagProbes = False

        if forceCoeffsPatches is not None:
            self.flagForceCoeffs = True
            self.forceCoeffsPatches = forceCoeffsPatches
            self.lRef = lRef
            self.ARef = ARef
        else:
            self.flagForceCoeffs = False

        if boundaryPatches is not None:
            self.flagBoundaries = True
            self.boundaryPatches = boundaryPatches
            self.boundaryQuantities = np.array(boundaryQuantities)
            self.boundaryQuantitiesIndices = np.array(boundaryQuantitiesIndices)
            self.boundaryDimensions = np.array(boundaryDimensions)

            for i in range(len(boundaryPatches)):
                self.xVar.append(None)
                self.iVar.append(None)
                self.nVar.append(None)
                self.nSkip.append(list())
                for j in range(len(boundaryQuantities)):
                    self.nSkip[i].append(None)
                    if i == 0:
                        self.flagNSet.append(False)

            if boundaryLimits is not None:
                self.flagBoundaryLimits = True
                self.boundaryLimits = np.array(boundaryLimits)
                if boundaryLimitsReduced is not None:
                    self.boundaryLimitsReduced = np.array(boundaryLimitsReduced)
                else:
                    self.boundaryLimitsReduced = np.array(boundaryLimits)
            else:
                self.flagBoundaryLimits = False
        else:
            self.flagBoundaries = False

    def setN(self, fileName, iVar, iBnd):
        fin = open(fileName, 'r')
        nSkip = 0
        while True:
            line = fin.readline()
            nSkip += 1
            if line.find(self.boundaryPatches[iVar]) >= 0:
                for i in range(4):
                    line = fin.readline()
                nSkip += 5
                self.nSkip[iBnd][iVar] = nSkip

                nVar = int(line[:-1])
                dL = (self.boundaryLimits[iBnd][1] - self.boundaryLimits[iBnd][0]) / float(nVar)
                xVar = np.linspace(self.boundaryLimits[iBnd][0] + dL, self.boundaryLimits[iBnd][1] - dL, nVar)
                iVarReduced = [i for i in range(nVar) if (self.boundaryLimitsReduced[iBnd][0] <= xVar[i] <=
                                                          self.boundaryLimitsReduced[iBnd][1])]
                self.nVar[iBnd] = len(iVarReduced)
                self.iVar[iBnd] = np.array(iVarReduced)
                self.xVar[iBnd] = xVar[self.iVar[iBnd]]

                fin.close()
                return


def reformatLines(fi):
    for line in fi:
        line = line.replace('(', '')
        line = line.replace(')', '')
        yield line
