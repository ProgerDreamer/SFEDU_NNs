import numpy as np
import matplotlib.pyplot as plt


LABEL_TO_COLOR = {0: 'r', 1: 'b', 2: 'purple',
                  3: 'y', 4: 'black'}


def plot_linear_boundaries(clsfr, ax, xlim, ylim,
                           grid_resolution=400, alpha=0.5):
    assert clsfr.input_dim == 2

    X1, X2 = np.meshgrid(np.linspace(*xlim, grid_resolution),
                         np.linspace(*ylim, grid_resolution))
    data = np.hstack((X1.ravel().reshape(-1, 1), X2.ravel().reshape(-1, 1)))
    Y_pred = np.apply_along_axis(lambda x: clsfr.predict(x), 1, data)

    levels = np.arange(-0.5, clsfr.num_classes, 1.)
    colors = [LABEL_TO_COLOR[i] for i in range(clsfr.num_classes)]
    ax.contourf(X1, X2,
                Y_pred.reshape(grid_resolution, grid_resolution),
                levels=levels, colors=colors, alpha=alpha)


def basic_scatter(ax, x, y, color=None, xlim=None, ylim=None):
    ax.scatter(x, y, color=color)
    if xlim is None:
        xlim = (None, ) * 2
    if ylim is None:
        ylim = (None, ) * 2
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


