import numpy as np
import scipy.optimize as opt
from util import logistic, calc_loss

class LogisticRegiression:

    def __init__(self, input_feature_size):
        self.input_size = input_feature_size;
        self.weight = np.zeros((1, input_feature_size))
        self.weight.shape = (1, input_feature_size)
        self.bias = 0

    def predict_prob(self, x):
        return logistic(self.weight.dot(x) + self.bias)

    def predict_class(self, x):
        prob = self.predict_prob(x)
        predict_class = np.zeros(prob.shape)
        _, size = x.shape
        for i in range(size):
            if prob[-i] >= 0.5:
                predict_class[i] = 1
        return predict_class

    def train(self, x, y, lr=0.01, max_epoch=10000, batch_size=10):
        feature_size, data_size = x.shape

        for epoch in range(max_epoch):
            for batch in range(batch_size):
                start_pos = int((batch) / 10 * data_size)
                end_pos = int((batch + 1) / 10 * data_size)
                y_pred_prob = self.predict_prob(x[:, start_pos:(end_pos - 1)])
                gradient_w = np.zeros((1, self.input_size))
                gradient_w = (y_pred_prob - y[:, start_pos:(end_pos - 1)]).dot(x[:, start_pos:(end_pos - 1)].transpose()) / data_size
                gradient_b = np.mean((y_pred_prob - y[:, start_pos:(end_pos - 1)]))
                self.weight -= gradient_w * lr
                self.bias -= gradient_b * lr
            loss = calc_loss(y_pred_prob, y[:, start_pos:(end_pos - 1)])
            if epoch % 100 == 0:
                print('The {}-th epoch, loss is {}'.format(epoch, loss))

    def __calc_loss(self, theta, x, y):
        y_pred = logistic(x.dot(theta))
        return calc_loss(y_pred, y)

    def __calc_gradient(self, theta, x, y):
        y_pred = logistic(x.dot(theta))
        _, data_size = x.shape
        return ((y_pred - y).dot(x)).transpose() / data_size

    def get_weight(self):
        return (self.weight, self.bias)

    def lib_train_method(self, x, y):
        _, data_size = x.shape
        ones = np.ones((1, data_size))
        new_x = np.r_[x, ones].transpose()
        theta = np.zeros(_ + 1)
        res = opt.minimize(fun=self.__calc_loss, x0=theta, args=(new_x, y), method='BFGS',jac=self.__calc_gradient)
        self.weight = res.x[0:2]
        self.bias = res.x[2]
