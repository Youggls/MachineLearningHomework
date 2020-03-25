import numpy as np
from util import sigmoid

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__output_size = output_size
        self.__weight1 = np.random.rand(hidden_size, input_size)
        self.__weight2 = np.random.rand(output_size, hidden_size)

    def set_weight(self, weight1, weight2):
        self.__weight1 = weight1
        self.__weight2 = weight2

    def predict_class(self, x):
        proba = self.forward(x)
        return np.argmax(proba, axis=1) + 1

    def forward(self, x):
        x = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)
        z2 = x.dot(self.__weight1.transpose())
        z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
        a2 = sigmoid(z2)
        z3 = a2.dot(self.__weight2.transpose())
        a3 = sigmoid(z3)
        return a3
