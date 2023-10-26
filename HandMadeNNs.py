import numpy as np

from util import softmax, sigmoid, tanh


class BinaryClassifier:
    def __init__(self, input_dim, lr=1.e-1):
        self.w = np.random.normal(size=input_dim)
        self.b = 0.
        self.input_dim = input_dim
        self.lr = lr

    def forward(self, x):
        return sigmoid(np.dot(self.w, x) + self.b)

    def backward(self, x, y_true):
        logit = self.forward(x)
        dw = self.lr * 2 * (y_true - logit) * logit * (1 - logit) * x
        db = self.lr * 2 * (y_true - logit) * logit * (1 - logit)
        self.w += dw
        self.b += db

    def predict(self, x):
        return 1 if self.forward(x) > 0.5 else 0

    def get_parameters(self):
        return self.w, self.b


class PerceptronClassifier:

    def __init__(self, input_dim, num_classes, lr=1.e-1):
        self.w = np.random.normal(size=(input_dim, num_classes))
        self.b = np.zeros(num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr

    def forward(self, x):
        return softmax(np.dot(x, self.w) + self.b)

    def backward(self, x, y_true):
        y_true_oh = np.zeros(self.num_classes, dtype='int')
        y_true_oh[y_true] = 1
        out = self.forward(x)
        d_w = np.dot(x.reshape(-1, 1), (y_true_oh - out).reshape(1, -1)) * self.lr
        d_b = (y_true_oh - out) * self.lr
        self.w += d_w
        self.b += d_b

    def predict(self, x):
        return np.argmax(np.dot(x, self.w) + self.b)

    def get_parameters(self):
        return self.w, self.b


class MLPClassifier:

    def __init__(self, input_dim, num_classes, hidden_sizes, lr=1.e-1):
        self.layer_sizes = (input_dim, *hidden_sizes, num_classes)
        self.weights = [np.random.normal(size=(dim1, dim2))
                        for dim1, dim2 in zip(self.layer_sizes[:-1],
                                              self.layer_sizes[1:])]
        self.biases = [np.zeros(dim2) for dim2 in self.layer_sizes[1:]]
        self.activation = sigmoid
        self.lr = lr

    def forward(self, x):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(np.dot(x, w) + b)
        w, b = self.weights[-1], self.biases[-1]
        return softmax(np.dot(x, w) + b)

    def backward(self, x, y_true):
        y_true_oh = np.zeros(self.layer_sizes[-1], dtype='int')
        y_true_oh[y_true] = 1

        left_store = [x]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            left_store.append(self.activation(np.dot(left_store[-1], w) + b))
        out = softmax(np.dot(left_store[-1], self.weights[-1]) + self.biases[-1])

        error = y_true_oh - out
        x = left_store[-1]
        d_w = np.dot(x.reshape(-1, 1), error.reshape(1, -1)) * self.lr
        d_b = error * self.lr
        self.weights[-1] += d_w
        self.biases[-1] += d_b

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            out = x
            x = left_store[i - 1]
            error = np.array([np.dot(self.weights[i][j, :], error) for j in range(self.layer_sizes[i])])
            error = error * out * (1 - out)
            d_w = np.dot(x.reshape(-1, 1), error.reshape(1, -1)) * self.lr
            d_b = error * self.lr
            self.weights[i - 1] += d_w
            self.biases[i - 1] += d_b

    def predict(self, x):
        return np.argmax(self.forward(x))

    def get_parameters(self):
        return self.weights, self.biases
