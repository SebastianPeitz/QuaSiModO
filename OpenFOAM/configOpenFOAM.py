def getPaths():
    pathOF = '/home/speitz/OpenFOAM/OpenFOAM-v1912/'  \
    # pathOF = '/cm/shared/apps/pc2/EB-SW/software/cae/OpenFOAM/6-foss-2018b/OpenFOAM-6/'

    pathThirdParty = '/home/speitz/OpenFOAM/ThirdParty-v1912/'

    additionalCommands = None
    # additionalCommands = ['module add cae/OpenFOAM/6-foss-2018b']

    return pathOF, pathThirdParty, additionalCommands


def getSolver():
    solver = 'pimpleFoam'
    return solver


def getSkipRows():
    skipRowsForceCoeffs = 13
    return skipRowsForceCoeffs


def getCourant():
    maxCo = 0.75
    return maxCo


def getPlatform():
    platform = 'linux64GccDPInt32Opt'
    platformThirdParty = 'linux64GccDPInt32'
    return platform, platformThirdParty
