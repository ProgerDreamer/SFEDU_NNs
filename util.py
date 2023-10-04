import numpy as np


def label_to_color(label: int):
    d = {0: 'r', 1: 'b', 2: 'purple',
         3: 'y', 4: 'b'}
    return d[label]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def accuracy(y_pred, y_true):
    return np.sum(y_true == y_pred) / y_true.size


def plot_decision_boundaries(one_layer_clsfr, ax):
    w, b = one_layer_clsfr.get_parameters()
    pass
