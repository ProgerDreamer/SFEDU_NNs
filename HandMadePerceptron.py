import numpy as np

from util import softmax


class PerceptronClassifier:

    def __init__(self, input_dim, num_classes, lr=1.e-1):
        self.w = np.random.normal((input_dim, num_classes))
        self.b = np.zeros(num_classes)
        self.lr = lr

    def forward(self, x):
        return softmax(np.dot(x, self.w) + self.b)

    def backward(self, x, y_true_oh):
        self.w += x.reshape(-1, 1) * (self.w * y_true_oh) * self.lr

    def predict(self, x):
        return np.argmax(np.dot(x, self.w) + self.b)
