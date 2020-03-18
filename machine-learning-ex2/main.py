from lr import LogisticRegiression
from util import read_data
from plot import plot_data
import numpy as np

if __name__ == '__main__':
    print('Start')
    # ex2data1
    data1 = read_data('./ex2/ex2data1.txt')
    x = data1[:, 0:2]
    x = x.transpose()
    y = data1[:, 2]
    y = y.transpose()
    # y.shape = (1, 100)
    lr = LogisticRegiression(2)
    lr.lib_train_method(x, y)
    weight = lr.get_weight()
    print('The weight and bias is: {}'.format(weight))
    plot_data(x, y, weight[0], weight[1])
    print('End')
