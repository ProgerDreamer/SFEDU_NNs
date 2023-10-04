import random
import numpy as np
import matplotlib.pyplot as plt

from HandMadeNNs import OneLayerClassifier
from util import label_to_color, accuracy


def lab_2_1():

    def predict(row, weights, delta):
        activation = delta
        for i in range(len(row) - 1):
            activation += weights[i] * row[i]
        return 1.0 if activation >= 0.0 else 0.0

    def train_we(train, lr, n_epoch):
        weights = [0.0 for i in range(len(train[0]) - 1)]

        for epoch in range(n_epoch):
            delta = 0.0

            sum_error = 0.0
            for row in train:
                prediction = predict(row, weights, delta)
                error = row[-1] - prediction
                sum_error += error ** 2
                delta = delta + lr * error
                for i in range(len(row) - 1):
                    weights[i] += lr * error * row[i]

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

        return weights, delta

    N = 10
    dataset = [[0] * 4] * 10
    x1 = np.random.random(N)
    x2 = x1 + [np.random.randint(10) / 10 for i in range(N)]
    x3 = np.random.random(N)
    x4 = x3 - [np.random.randint(10) / 10 for i in range(N)] - 0.1
    for i in range(10):
        if i < 5:
            dataset[i] = [1, x1[i], x2[i], 0]
        else:
            dataset[i] = [1, x3[i], x4[i], 1]
    l_rate = 0.1
    n_epoch = 1000
    weights, delta = train_we(dataset, l_rate, n_epoch)
    print(weights)
    for row in dataset:
        predicted = predict(row, weights, delta)
        print('Выбрано:%d , Предсказано:%d ' % (row[-1], predicted))
    for row in dataset:
        if row[-1] == 1:
            plt.scatter(row[1], row[2], s=10, c='red')
        else:
            plt.scatter(row[1], row[2], s=10, c='blue')

    def line(x1, weights):
        return ((weights[1] * x1) / -weights[2]) - (weights[0] / weights[2])

    x1_range = np.arange(-5, 5, 0.5)
    plt.plot(x1_range, line(x1_range, weights), color='black')
    plt.grid(True)
    plt.show()


def lab_2_2():
    # 3 classes

    # data generation
    support_points = np.array([[5, 0], [0, 3], [-2, -4]])
    points_around = 40

    X = np.vstack([np.random.normal(size=(points_around, 2)) + p for p in support_points])
    Y = np.vstack([np.full((points_around, 1), i + 1, dtype='int') for i in range(support_points.shape[0])])
    labels = [i + 1 for i in range(np.unique(Y).size)]
    assert X.shape[0] == Y.shape[0]

    plt.scatter(X[:, 0], X[:, 1], color=[label_to_color(y[0]) for y in Y])
    plt.show()

    # training
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    split = 0.2

    split = int((1 - split) * X.shape[0])
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    clsfr = OneLayerClassifier(X.shape[1], len(labels), lr=1.e-2)

    epochs = 10
    for i in range(epochs):
        # train cycle
        for x, y in zip(X_train, Y_train):
            one_hot = np.zeros(len(labels), dtype='int')
            one_hot[labels.index(y)] = 1
            clsfr.backward(x, one_hot)

        # test_cycle
        Y_pred = np.zeros_like(Y_test)
        for j, (x, y) in enumerate(zip(X_test, Y_test)):
            Y_pred[j] = labels[clsfr.predict(x)]

        # logging
        parameters = clsfr.get_parameters()
        print(f'Epoch {i + 1} finished; accuracy: {accuracy(Y_pred, Y_test)}')
        print('Parameters;\n weights: \n', parameters[0], '\nbiases: ', parameters[1])
        print()


def main():
    np.random.seed(0)
    # lab_2_1()
    lab_2_2()
    pass


if __name__ == '__main__':
    main()
