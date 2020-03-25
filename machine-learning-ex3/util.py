import scipy.io as scio
import numpy as np

def read_matlab(path):
    return scio.loadmat(path)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic(x):
    ans = np.zeros(x.shape)
    size = 0
    if (len(x.shape)) == 1:
        size = x.shape[0]
        for i in range(size):
            if x[i] >= 0:
                ans[i] = 1 / (1 + np.exp(-x[i]))
            else:
                ans[i] = np.exp(x[i]) / (1 + np.exp(x[i]))
    else:
        a, b = x.shape
        for i in range(a):
            for j in range(b):
                if x[i, j] >= 0:
                    ans[i, j] = 1 / (1 + np.exp(-x[i, j]))
                else:
                    ans[i, j] = np.exp(x[i, j]) / (1 + np.exp(x[i, j]))
    return ans

def trans_label_to_onehot(label):
    length, _ = label.shape
    res = np.zeros((length, 10))
    for i in range(length):
        res[i, label[i,0] - 1] = 1
    return res

def calc_loss(y_true, y_pred, weight, l=0.1):
    """
    y_true shoule be n * 1, range is [1, 10] integer
    y_pred is n * 10, onehot vector
    """
    if y_pred.shape != y_true.shape:
        raise ValueError('Shape Not Same!')
    size = 0
    if len(y_pred.shape) == 1:
        size = y_pred.shape[0]
        y_pred.shape = (1, size)
        y_true.shape = (1, size)
        weight_size = weight.shape[0]
        weight.shape = (1, weight_size)
    else:
        _, size = y_pred.shape
    # return (1 / size) * np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred), axis=1) + l * np.sum(weight ** 2, axis=1)
    return (1 / size) * np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred), axis=1)

def calc_precision(y_true, y_pred):
    """
    y_true and y_pred must be n * 1
    """
    total, _ = y_true.shape
    TP = 0
    for i in range(total):
        if y_true[i, 0] == y_pred[i, 0]:
            TP += 1
    return TP/total * 100
