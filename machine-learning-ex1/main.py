import numpy as np
import matplotlib.pyplot as plt
from util import read_data, normalize
from plot import plot_data
from linear import LinearRegression

if __name__ == '__main__':
    print('Start')
    # One var regiression
    print('Start linear regression with one variable')
    data = read_data('./ex1/ex1data1.txt')
    linear = LinearRegression(1, 1, 0.01)
    x = np.mat(data[:,0])
    y = np.mat(data[:,1])
    print('Start train model linear regression with variable')
    linear.train(x, y, 1000,  lr=0.01)
    print('End train model linear regireesion with one variable')
    x_list = x.tolist()[0]
    x_list.sort()
    x_sorted = np.mat([x_list])
    y_pred = linear.predict(x_sorted)
    print('End linear regression with one variable')
    # multiple var regressioin
    print('Start linear regression with multiple variables')
    data_multiple = read_data('./ex1/ex1data2.txt')
    # normalize data
    data_multiple = normalize(data_multiple)
    linear2 = LinearRegression(2, 1, 0.01)
    x2 = np.mat(data_multiple[:,0:2])
    y2 = np.mat(data_multiple[:, 2])
    lr_list = [0.0001, 0.001, 0.01, 1/np.exp(1)]
    loss = []
    for lr in lr_list:
        print('Start linear regression with multiple variables train when learning rate={}'.format(lr))
        loss.append(linear2.train(x2.transpose(), y2, lr=lr, max_turns=50))
        print('End linear regression with multiple variables train when learning rate={}'.format(lr))
    print('End linear regression with multiple variables')
    # Plot data
    plot_data(x, y, x_sorted, y_pred, lr_list, loss)
    print('End')
