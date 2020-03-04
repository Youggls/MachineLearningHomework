import matplotlib.pyplot as plt
import numpy as np
import math

def plot_data(x, y, y_pred):
    # plt.subplot(2, 2, 1)
    print(np.min(x))
    # plt.xticks(np.arange(math.floor(np.min(x)), math.ceil(np.max(x)), 1))
    # plt.yticks(np.arange(math.floor(np.min(y)), math.ceil(np.max(y)), 5))
    plt.plot(x, y, 'r--')
    plt.legend()
    # plt.plot(x, y_pred, color='red')
    plt.show()
