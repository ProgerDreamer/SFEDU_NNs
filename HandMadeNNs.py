import numpy as np

from util import softmax, sigmoid


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
