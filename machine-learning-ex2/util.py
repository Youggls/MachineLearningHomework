import numpy as np

def read_data(path):
    file = open(path, mode='r', encoding='utf8')
    ans = []
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n')
        line_split = line.split(',')
        ans.append(list(map(float, line_split.copy())))
    return np.array(ans)

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
        _, size = x.shape
        for i in range(size):
            if x[0,i] >= 0:
                ans[0,i] = 1 / (1 + np.exp(-x[0,i]))
            else:
                ans[0,i] = np.exp(x[0,i]) / (1 + np.exp(x[0,i]))
    return ans

def calc_loss(y_pred, y_true):
    """
    y_pred and y_true must be 1 * n
    """
    if y_pred.shape != y_true.shape:
        raise ValueError('Shape Not Same!')
    size = 0
    if len(y_pred.shape) == 1:
        size = y_pred.shape[0]
    else:
        _, size = y_pred.shape
    return (1 / size) * np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
