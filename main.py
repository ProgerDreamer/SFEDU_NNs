import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

from sklearn.linear_model import Perceptron

from HandMadeNNs import PerceptronClassifier, BinaryClassifier
from util import *
from plot_util import *
from ml_util import *


def lab_2_1():
    import random

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
    # 4 classes

    # data generation
    check_points = np.array([[1, 5], [1, 0], [-1, 1], [-1, -1], [2, -2]])
    check_labels = np.array([0, 1, 2, 2, 3]).reshape(-1, 1)
    X, Y = gen_gauss_for_clsfr(check_points, labels=check_labels.ravel(), S=[0.3, 0.2, 0.4, 0.4, 0.3],
                               gen_for_each=50)
    check_points = check_points[1:4]
    check_labels = check_labels[1:4]
    labels = np.unique(Y)
    # plt.scatter(X[:, 0], X[:, 1], color=[label_to_color(y[0]) for y in Y])
    # plt.show()

    # training
    X_train, X_test, Y_train, Y_test = split_and_shuffle(X, Y)

    # show training set
    fig, ax = plt.subplots()
    basic_scatter(ax, X_train[:, 0], X_train[:, 1], [LABEL_TO_COLOR[y[0]] for y in Y_train],
                  xlim=(-3., 3.), ylim=(-4., 6.5))
    fig.savefig('./lab2/train_set.png')

    clsfr = PerceptronClassifier(X.shape[1], len(labels), lr=1.e-3)

    def log_proc(clsfr, x_test, y_test, epoch_num):
        Y_pred = np.zeros_like(y_test)
        for j, (x, y) in enumerate(zip(x_test, y_test)):
            Y_pred[j] = clsfr.predict(x)
        parameters = clsfr.get_parameters()
        if epoch_num == 0:
            progress_id = 'Not trained'
        else:
            progress_id = f'Epoch {epoch_num + 1}'
        print(f'{progress_id}; accuracy: {accuracy(Y_pred, y_test)}')
        print('Parameters;\n weights: \n', parameters[0], '\nbiases: \n', parameters[1])
        print()

    def boundaries_proc(clsfr, x_test, y_test, epoch_num):
        # show test set with boundaries
        to_title = str(epoch_num)
        fig, ax = plt.subplots()
        plot_linear_boundaries(clsfr, ax, [-3., 3.], [-4., 6.5])
        ax.scatter(x_test[:, 0], x_test[:, 1], color=[LABEL_TO_COLOR[y[0]] for y in y_test])
        ax.scatter(check_points[:, 0], check_points[:, 1],
                   color=[LABEL_TO_COLOR[y[0]] for y in check_labels],
                   marker='*', s=40, edgecolors='black', linewidths=0.5)
        fig.savefig(f'./lab2/contourf_{to_title}.png')
        plt.close(fig)

    fit_cycle(clsfr, X_train, X_test, Y_train, Y_test, epochs=40,
              callbacks=(log_proc, boundaries_proc))

    # skl perceptron
    clsfr = Perceptron()
    clsfr.fit(X_train, np.squeeze(Y_train))
    score = clsfr.score(X_test, np.squeeze(Y_test))
    print(f'Sklearn perceptron score for comparison: {score}')


def lab_3():
    base_points = np.array([[1, 5], [1, 0], [-1, 1], [-1, -1], [2, -2]])
    base_labels = np.array([0, 1, 2, 2, 3]).reshape(-1, 1)
    labels = np.unique(base_labels)

    epochs = 100
    x_data = np.arange(epochs + 1)

    for scale in (0.2, 0.5, 1., 2., 5.):
        X, Y = gen_gauss_for_clsfr(base_points * scale, labels=base_labels.ravel(),
                                   S=[0.3, 0.2, 0.4, 0.4, 0.3],
                                   gen_for_each=50)

        fig, ax = plt.subplots()
        plot_scale = max(scale, 1.)
        basic_scatter(ax, X[:, 0], X[:, 1], [LABEL_TO_COLOR[y[0]] for y in Y],
                      xlim=np.array([-3., 3.]) * plot_scale, ylim=np.array([-4., 6.5]) * plot_scale)
        fig.savefig(f'./lab3/train_set_scale_{scale:.2f}.png')

        X_train, X_test, Y_train, Y_test = split_and_shuffle(X, Y)

        for lr in (1.e-3, 1.e-2, 1.e-1, 1.):
            clsfr = PerceptronClassifier(X.shape[1], len(labels), lr=lr)
            learning_curve = fit_cycle(clsfr, X_train, X_test, Y_train, Y_test,
                                       metrics=(('accuracy', accuracy), ), verbose=True)[:, 0]
            fig, ax = plt.subplots()
            ax.plot(x_data, learning_curve)
            ax.set_title('learning progress\n'
                         f'maximum accuracy achieved: {np.max(learning_curve):.3f}')
            ax.set_xlabel('epoch')
            ax.set_ylabel('accuracy')
            ax.set_ylim(0., 1.1)
            fig.savefig(f'./lab3/learning_curve_scale{scale:.2f}_lr{lr:.3g}.png')
            plt.close(fig)

            df = pd.DataFrame()
            df['accuracy'] = learning_curve
            df.to_csv(f'./lab3/learning_curve_scale{scale:.2f}_lr{lr:.3g}.csv',
                      index=False)


# def lab_2_2_perceptron():
#     # binary classification
#
#     # data generation
#     support_points = np.array([[5, 0], [0, 3]])
#     points_around = 100
#
#     X = np.vstack([np.random.normal(size=(points_around, 2)) + p for p in support_points])
#     Y = np.vstack([np.full((points_around, 1), i, dtype='int') for i in range(support_points.shape[0])])
#     assert X.shape[0] == Y.shape[0]
#
#     fig, ax = plt.subplots()
#     ax.scatter(X[:, 0], X[:, 1], color=[label_to_color(y[0]) for y in Y])
#     fig.savefig('./lab1/data.png')
#     plt.close(fig)
#
#     # training
#     idx = np.arange(X.shape[0])
#     np.random.shuffle(idx)
#     split = 0.2
#
#     split = int((1 - split) * X.shape[0])
#     train_idx, test_idx = idx[:split], idx[split:]
#     X_train, Y_train = X[train_idx], Y[train_idx]
#     X_test, Y_test = X[test_idx], Y[test_idx]
#
#     clsfr = BinaryClassifier(X.shape[1], lr=1.e-2)
#
#     epochs = 10
#     for i in range(epochs):
#         # train cycle
#         for x, y in zip(X_train, Y_train):
#             y_pred = clsfr.forward(x)
#             clsfr.backward(x, y, y_pred)
#
#         # test_cycle
#         Y_pred = np.zeros_like(Y_test)
#         for j, (x, y) in enumerate(zip(X_test, Y_test)):
#             Y_pred[j] = clsfr.predict(x)
#
#         # logging
#         parameters = clsfr.get_parameters()
#         print(f'Epoch {i + 1} finished; accuracy: {accuracy(Y_pred, Y_test)}')
#         print('Parameters;\n weights: \n', parameters[0], '\nbiases: ', parameters[1])
#         print()
#
#     fig, ax = plt.subplots()
#     plot_linear_boundaries(clsfr, ax, [-4, 8], [-4, 10],
#                            colors=[label_to_color(0), label_to_color(1)])
#     ax.scatter(X_train[:, 0], X_train[:, 1], color=[label_to_color(y[0]) for y in Y_train])
#     fig.savefig('./lab1/train_with_boundaries.png')
#
#     fig, ax = plt.subplots()
#     plot_linear_boundaries(clsfr, ax, [-4, 8], [-4, 10],
#                            colors=[label_to_color(0), label_to_color(1)])
#     ax.scatter(X_test[:, 0], X_test[:, 1], color=[label_to_color(y[0]) for y in Y_test])
#     fig.savefig('./lab1/test_with_boundaries.png')


def main():
    np.random.seed(200)
    # lab_2_1()
    # lab_2_2()
    # lab_2_2_perceptron()
    lab_3()
    pass


if __name__ == '__main__':
    main()
