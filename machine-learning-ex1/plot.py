import matplotlib.pyplot as plt
import numpy as np
import math
from util import squre_loss
from mpl_toolkits.mplot3d import Axes3D

def plot_data(x, y, x_sorted, y_pred, lr_list, loss_list):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222, projection='3d')
    ax2 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    x_sorted = np.array(x_sorted)
    y_pred = np.array(y_pred)
    x_sorted.shape = (97,)
    y_pred.shape = (97,)
    ax1.scatter([x], [y], c='r')
    ax1.plot(x_sorted, y_pred, color='blue')
    theta1 = np.arange(-1, 4, 0.1)
    theta0 = np.arange(-10, 10, 0.1)
    loss = np.zeros((200, 50))
    theta0, theta1 = np.meshgrid(theta1, theta0)
    for i in range(0, 200):
        for j in range(0, 50):
            y_pred_list = theta0[i][j] * x + theta1[i][j]
            ans = squre_loss(y_pred_list, y)
            loss[i, j] = ans
    ax2.contour(theta0, theta1, loss,  np.logspace(-2, 3, 20), cmap=plt.cm.jet)
    ax3.plot_surface(theta0, theta1, loss, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
    ax3.set_zlabel('Cost')
    ax3.set_xlabel('theta1')
    ax3.set_ylabel('theta0')
    ax2.set_xlabel('theta1')
    ax2.set_ylabel('theta0')
    for i in range(len(lr_list)):
        ax4.plot(np.arange(0, len(loss_list[i]), 1), np.array(loss_list[i]), label=lr_list[i])
    ax4.set_xlabel('epoch', fontsize=18)
    ax4.set_ylabel('cost', fontsize=18)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax4.set_title('Lerining rate')
    plt.show()
