import numpy as np
import pickle
import scipy.optimize as opt
from util import logistic, sigmoid, calc_loss

def load_lr(path):
    f = open(path, 'rb')
    return pickle.load(f)

class LogisticRegression:

    def __init__(self, feature_size, class_number):
        self.__feature_size = feature_size
        self.__class_numer = class_number
        self.__weight = np.random.rand(self.__class_numer, self.__feature_size)
        self.__bias = np.random.rand(self.__class_numer, 1)
    
    def __predict_single_class(self, x, class_index):
        return logistic(self.__weight[class_index].dot(x) + self.__bias[class_index])

    def predict(self, x):
        return logistic(self.__weight.dot(x) + self.__bias)

    def predict_class(self, x):
        feature_size, data_size = x.shape
        proba = self.predict(x)
        ans = np.zeros((data_size, 1))
        for i in range(data_size):
            max_index = np.argmax(proba[:, i])
            ans[i, 0] = max_index + 1
        return ans
    
    def save_model(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f)

    def train(self, x, label, lr=0.01, l=0.1, batch_size=100, max_turns=500):
        for i in range(0, self.__class_numer):
            print('Begin to train class {}.'.format(i))
            self.__train(x, label[i], i, lr, l, batch_size, max_turns)

    def __loss(self, theta, X, y):
        ''' cost fn is -l(theta) for you to minimize'''
        return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

    def __regularized_loss(self, theta, X, y, l=1):
        '''you don't penalize theta_0'''
        theta_j1_to_n = theta[1:]
        regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

        return self.__loss(theta, X, y) + regularized_term

    def __gradient(self, theta, X, y):
        '''just 1 batch gradient'''
        return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

    def __regularized_gradient(self, theta, X, y, l=1):
        '''still, leave theta_0 alone'''
        theta_j1_to_n = theta[1:]
        regularized_theta = (l / len(X)) * theta_j1_to_n

        # by doing this, no offset is on theta_0
        regularized_term = np.concatenate([np.array([0]), regularized_theta])

        return self.__gradient(theta, X, y) + regularized_term

    def __lib_train(self, X, y, l=1):
        X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
        # init theta
        theta = np.zeros(X.shape[1])

        # train it
        res = opt.minimize(fun=self.__regularized_loss,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=self.__regularized_gradient,
                       options={'disp': True})
        # get trained parameters
        final_theta = res.x

        return final_theta

    def lib_train(self, X, y, l=1):
        k_theta = np.array([self.__lib_train(X, y[k]) for k in range(10)])
        self.__bias = k_theta[:, 0]
        self.__weight = k_theta[:, 1:402]
        self.__bias.shape = (self.__class_numer, 1)

    def __train(self, x, label, class_index, lr, l, batch_size, max_turns):
        """
        :x is all the train data
        :label is the label, size is n * 1
        """
        feature_size, data_size = x.shape
        for turn in range(max_turns):
            for batch in range(int(data_size / batch_size)):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                batch_x = x[:, start:end]
                batch_l = label[start:end]
                batch_predict = self.__predict_single_class(batch_x, class_index)
                delta = batch_predict - batch_l
                gradient_w = delta.dot(batch_x.transpose()) / data_size
                # gradient_w = delta.dot(batch_x.transpose()) / data_size + l / data_size * np.sum(self.__weight[class_index])
                gradient_b = np.sum(delta) / data_size
                self.__weight[class_index] -= lr * gradient_w
                self.__bias[class_index] -= lr * gradient_b
            if turn % 100 == 0:
                loss = calc_loss(batch_l, batch_predict, self.__weight[class_index], l)
                print('The {}-th turn\' loss is {}'.format(turn, loss))


