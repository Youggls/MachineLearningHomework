import matplotlib.pyplot as plt
import numpy as np

def plot_sub(X, y, ax):
    pos = np.nonzero(y == 1)
    neg = np.nonzero(y == 0)
    ax.plot(X[pos, 0], X[pos, 1], linestyle='', marker='+', color='k')
    ax.plot(X[neg, 0], X[neg, 1], linestyle='', marker='o', color='y')

def plot_boundary(X, y, clf, ax):
    x1_plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2_plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1_plot, x2_plot)
    vals = np.zeros(X1.shape)

    for i in range(X1.shape[1]):
        X_tmp = np.hstack((X1[:, i:i + 1], X2[:, i:i + 1]))
        vals[:, i] = clf.predict(X_tmp)
    ax.contour(X1, X2, vals, levels=[0])

def plot_boundary_linear(X, y, clf, ax):
    coef = clf.coef_.ravel()
    intercept = clf.intercept_.ravel()

    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = -1.0 * (coef[0] * xp + intercept[0]) / coef[1]

    ax.plot(xp, yp, linestyle='-', color='b')

def plot(x1, y1, clf1, x2, y2, clf2, x3, y3, clf3):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    plot_sub(x1, y1, ax1)
    plot_boundary_linear(x1, y1, clf1, ax1)
    plot_sub(x2, y2, ax2)
    plot_boundary(x2, y2, clf2, ax2)
    plot_sub(x3, y3, ax3)
    plot_boundary(x3, y3, clf3, ax3)
    plt.show()
