import numpy as np
from util import squre_loss
class LinearRegression:

    def __init__(self, input_feature_num, output_feature_num, lr=0.01):
        self.__input_size = input_feature_num
        self.__output_size = output_feature_num
        self.__learning_rate = lr
        # Weight matrix, size is output * input
        self.__Weight = np.mat(np.random.rand(self.__output_size, self.__input_size))
        # Bias vector, size is output * 1
        self.__bias = np.mat(np.random.rand(self.__output_size, 1))

    def predict(self, x):
        return self.__Weight * x + self.__bias

    def train(self, x, y, max_turns=10):
        """
        :x is the input, shape must be data_size * feature_size
        :y is the output, shape must be data_size * output_size
        :max_turns is the max training turns. Default value is 10.
        """
        x = np.mat(x)
        y = np.mat(y)
        data_size, _ = x.shape
        for turn in range(max_turns):
            y_pred = self.predict(x)
            gradient_w = np.mat(np.zeros((self.__output_size, self.__input_size)))
            gradient_b = np.mat(np.zeros((self.__output_size, 1)))
            for index in range(data_size):
                gradient_w += (y_pred[:, index] - y[:, index]) * x[:, index].transpose()
                gradient_b += (y_pred[:, index] - y[:, index])
            self.__Weight -= (1.0/data_size) * self.__learning_rate * gradient_w
            self.__bias -= (1.0/data_size) * self.__learning_rate * gradient_b
            loss = squre_loss(y_pred, y)
            print('The {}-th turns: the loss is {}.'.format(turn, loss))
        print('Training stop.')
