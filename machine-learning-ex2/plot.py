import matplotlib.pyplot as plt
import numpy as np

def plot_data(x, y, theta, bias):
    y.shape = (100,)
    theta.shape = (2,)
    for i in range(0, 100):
        if y[i] == 1:
            plt.scatter(x[0, i], x[1, i], c='green')
        else:
            plt.scatter(x[0, i], x[1, i], c='yellow')
    xx = np.arange(0, 100, 1)
    yy = -theta[0]/theta[1] * xx - bias / theta[1]
    plt.plot(xx, yy)
    plt.show()