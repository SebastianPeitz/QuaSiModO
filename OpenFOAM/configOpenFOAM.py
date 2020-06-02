def getPaths():
    pathOF = '/home/speitz/OpenFOAM/OpenFOAM-v1912/'
    pathThirdParty = '/home/speitz/OpenFOAM/ThirdParty-v1912/'

    return pathOF, pathThirdParty


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
