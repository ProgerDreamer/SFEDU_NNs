import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def label_to_color(label: int):
    d = {0: 'r', 1: 'b', 2: 'purple',
         3: 'y', 4: 'b'}
    return d[label]
