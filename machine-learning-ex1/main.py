import numpy as np
import matplotlib as plt
from util import read_data
from plot import plot_data
from linear import LinearRegression
if __name__ == '__main__':
    print('Start')
    data = read_data('./ex1/ex1data1.txt')
    # plot_data(data[:, 0], data[:, 1])
    linear = LinearRegression(1, 1)
    x = np.mat(data[:,0])
    y = np.mat(data[:,1])
    linear.train(x, y, 10)
    y_pred = linear.predict(x)
    plot_data(np.array(x), np.array(y), np.array(y_pred))
    print('End')
