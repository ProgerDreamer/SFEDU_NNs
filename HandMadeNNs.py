import numpy as np

from util import softmax


class OneLayerClassifier:

    def __init__(self, input_dim, num_classes, lr=1.e-1):
        self.w = np.random.normal(size=(input_dim, num_classes))
        self.b = np.zeros(num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr

    def forward(self, x):
        return softmax(np.dot(x, self.w) + self.b)

    def backward(self, x, y_true_oh):
        d_w = np.dot(x.reshape(-1, 1), y_true_oh.reshape(1, -1)) * self.lr
        d_b = y_true_oh * self.lr
        self.w += d_w
        self.b += d_b

    def predict(self, x):
        return np.argmax(np.dot(x, self.w) + self.b)

    def get_parameters(self):
        return self.w, self.b
