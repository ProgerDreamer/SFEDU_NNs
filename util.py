from collections.abc import Iterable

import numpy as np


def sigmoid(x):
    z = np.exp(x)
    return z / (1 + z)


def tanh(x):
    z = np.exp(x)
    return (z + 1. / z) / (z - 1. / z)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def accuracy(y_pred, y_true):
    return np.sum(y_true == y_pred) / y_true.size


def gen_gauss_data(mus, gen_for_each=100, S=1.):
    if not isinstance(mus, np.ndarray):
        mus = np.array(mus)
    mus = mus.astype(np.float32)
    if isinstance(gen_for_each, int):
        gen_for_each = [gen_for_each] * len(mus)

    if not isinstance(S, Iterable):
        S = np.array([S] * len(mus))
    assert len(S) == len(mus)

    data = np.vstack([np.tile(mus[i], (gen_for_each[i], 1))
                      for i, _ in enumerate(gen_for_each)])
    S = np.vstack([np.full((reps, 1), s) for reps, s in zip(gen_for_each, S)])
    gauss_noise = np.random.normal(size=data.shape)
    data = data + gauss_noise * S

    return data


def gen_gauss_for_clsfr(mus, gen_for_each=100, S=1., labels=None):
    if isinstance(gen_for_each, int):
        gen_for_each = [gen_for_each] * len(mus)
    if labels is None:
        labels = [i for i, _ in enumerate(gen_for_each)]

    X = gen_gauss_data(mus, gen_for_each, S)
    Y = np.vstack([np.array([l for _ in range(reps)]).reshape(-1, 1)
                   for l, reps in zip(labels, gen_for_each)])
    return X, Y


def get_nonlinear_clsfn_data():
    X1 = np.linspace()
    X2 = np.linspace()


def split_and_shuffle(X, Y, test_part=0.2):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    test_part = int((1 - test_part) * X.shape[0])
    train_idx, test_idx = idx[:test_part], idx[test_part:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    return X_train, X_test, Y_train, Y_test

