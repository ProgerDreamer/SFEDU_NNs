import numpy as np
import matplotlib.pyplot as plt


def label_to_color(label: int):
    d = {0: 'r', 1: 'b', 2: 'purple',
         3: 'y', 4: 'b'}
    return d[label]


def plot_linear_boundaries(one_layer_clsfr, ax, xlim, ylim,
                           colors, alpha=0.5):
    w, b = one_layer_clsfr.get_parameters()
    assert w.shape == (2, )
    assert len(colors) == 2

    def line(x):
        return (b - w[0] * x) / w[1]

    X = np.linspace(*xlim, 1000)
    Y = line(X)

    ax.fill_between(X, ylim[0], Y, facecolor=colors[0], alpha=alpha)
    ax.fill_between(X, Y, ylim[1], facecolor=colors[1], alpha=alpha)

