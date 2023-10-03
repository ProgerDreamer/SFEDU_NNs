import random
import numpy as np
import matplotlib.pyplot as plt

from HandMadePerceptron import PerceptronClassifier
from util import label_to_color


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
    assert X.shape[0] == Y.shape[0]

    plt.scatter(X[:, 0], X[:, 1], color=[label_to_color(y[0]) for y in Y])
    plt.show()

    # training


def main():
    np.random.seed(0)
    # lab_2_1()
    lab_2_2()
    pass


if __name__ == '__main__':
    main()
