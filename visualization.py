import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(showPlot=True, fOut=None, **kwargs):
    """
    multiPlotLines2D

    Function for 2D plotting of data versus time. Each data set needs to be specified within kwargs (e.g., y=data.y)
    """

    class structPlot:

        flagRef = False
        markerSize = 0
        marker = None
        iplot = 0
        type = 'Lines'
        x = []
        phase = False
        flagLegend = False

        def __init__(self, value_, key_):
            self.t = value_['t']
            self.y = value_[key_]
            self.label = key_
            if 'markerSize' in value_.keys():
                self.markerSize = value_['markerSize']
                self.marker = 'o'

            if 'reference' in value_.keys():
                self.ref = value_['reference']
                self.flagRef = True

            if 'iplot' in value_.keys():
                self.iplot = value_['iplot']

            if 'type' in value_.keys():
                self.type = value_['type']

            if 'x' in value_.keys():
                self.x = value_['x']

            if 'phase' in value_.keys():
                self.phase = value_['phase']

            if 'legend' in value_.keys():
                self.flagLegend = value_['legend']

    nPlots = 1
    plots = []
    for key, value in kwargs.items():
        plots.append(structPlot(value, key))
        if plots[-1].iplot + 1 > nPlots:
            nPlots = plots[-1].iplot + 1

    if nPlots == 1:
        ix = 1
        iy = 1
    elif nPlots == 2:
        ix = 2
        iy = 1
    elif nPlots == 3:
        ix = 2
        iy = 2
    elif nPlots == 4:
        ix = 2
        iy = 2
    elif nPlots == 5:
        ix = 3
        iy = 2
    elif nPlots == 6:
        ix = 3
        iy = 2
    elif nPlots == 7:
        ix = 3
        iy = 3
    elif nPlots == 8:
        ix = 3
        iy = 3
    elif nPlots == 9:
        ix = 3
        iy = 3

    fig = plt.figure(figsize=[12.80, 7.68])

    axes, labels = list(), list()
    for i in range(nPlots):
        axes.append([])
        labels.append('')

    ni = np.zeros([nPlots], dtype=int)

    for s in range(len(plots)):

        ip = plots[s].iplot
        if ni[ip] == 0:
            if plots[s].type == 'Surface':
                axes[ip] = fig.add_subplot(ix, iy, ip + 1, projection='3d')
                axes[ip].set_zlabel(plots[ip].label)
            else:
                axes[ip] = fig.add_subplot(ix, iy, ip + 1)
        # else:
        #     axes[ip] = plt.subplot(ix, iy, ip + 1)
        # plt.subplot(axes[ip])

        if ni[ip] == 0:
            lineStyle = 'solid'
        elif ni[ip] == 1:
            lineStyle = 'dashed'
        elif ni[ip] == 2:
            lineStyle = 'dotted'
        elif ni[ip] == 3:
            lineStyle = 'dashdot'

        ni[ip] += 1

        if len(plots[s].y.shape) == 1:
            y = np.zeros([plots[s].y.shape[0], 1], dtype=float)
            y[:, 0] = plots[s].y
            plots[s].y = y

        for i in range(plots[s].y.shape[1]):
            if plots[s].type == 'Lines':
                iCol = float(i) / float(np.maximum(plots[s].y.shape[1] - 1, 1))
                if plots[s].y.shape[1] > 1:
                    labelString = plots[s].label + str(i)
                else:
                    labelString = plots[s].label
                if plots[s].phase:

                    axes[ip].plot(plots[s].y[:, 0], plots[s].y[:, 1], color=plt.cm.RdYlBu(iCol), linestyle=lineStyle,
                                  linewidth=2, marker=plots[s].marker, markersize=plots[ip].markerSize,
                                  label=labelString)
                else:
                    axes[ip].plot(plots[s].t, plots[s].y[:, i], color=plt.cm.RdYlBu(iCol), linestyle=lineStyle,
                                  linewidth=2, marker=plots[s].marker, markersize=plots[ip].markerSize,
                                  label=labelString)
                    if plots[s].flagRef:
                        for j in range(len(plots[s].ref.iRef)):
                            iCol = float(plots[s].ref.iRef[j]) / float(np.maximum(plots[s].y.shape[1] - 1, 1))
                            # axes[ip].plot(plots[s].t, plots[s].ref.z[:len(plots[s].t), plots[s].ref.iRef[j]].toarray(),
                            axes[ip].plot(plots[s].t, plots[s].ref.z[:len(plots[s].t), plots[s].ref.iRef[j]],
                                          color=plt.cm.RdYlBu(iCol), linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=2,
                                          label=labelString)
            elif plots[s].type == 'Surface':
                X = np.outer(plots[s].x, np.ones(len(plots[s].t)))
                T = np.outer(np.ones(len(plots[s].x)), plots[s].t)

                axes[ip].plot_surface(T, X, plots[s].y.T, cmap='viridis', edgecolor='none')

        plt.grid(True)

        labels[ip] = labels[ip] + plots[s].label + ' / '

    for ip in range(nPlots):
        axes[ip].set_xlabel('t')
        if plots[ip].type == 'Lines':
            axes[ip].set_ylabel(labels[ip][:-2])
            if plots[ip].flagLegend:
                axes[ip].legend()
        elif plots[ip].type == 'Surface':
            axes[ip].set_ylabel('x')

    if fOut is not None:
        plt.savefig(fOut)

    if showPlot:
        plt.show()

    return fig


def plotPhase2D(y, fOut=None, showPlot=True):
    fig = plt.figure()
    plt.plot(y[:, 0], y[:, 1])

    plt.grid(True)

    if fOut is not None:
        plt.savefig(fOut)

    if showPlot:
        plt.show()

    return fig


def surfacePlot(x, t, y, fOut=None, showPlot=True):
    X = np.outer(x, np.ones(len(t)))
    T = np.outer(np.ones(len(x)), t)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, y, cmap='viridis', edgecolor='none')
    # ax.set_title('Surface plot')

    plt.grid(True)
    plt.set_xlabel('t')
    plt.set_ylabel('x')
    plt.set_zlabel('y')

    if fOut is not None:
        plt.savefig(fOut)

    if showPlot:
        plt.show()

    return fig


def plotLines3D():
    pass


def plotPhase3D():
    pass
