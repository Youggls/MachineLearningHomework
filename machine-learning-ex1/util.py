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

def squre_loss(y_predict, y_true):
    """
    Squre loss function,
    :y_predict is the predict output value, the shape must be n * m
    :y_true is the excepted output value, the shape must be n * m
    """
    _, __ = y_predict.shape
    ___, ____ = y_true.shape
    y_predict = np.array(y_predict)
    y_true = np.array(y_true)
    if __ != ____:
        raise ValueError('Error shape of input!')
    elif _ != ___:
        raise ValueError('The input size not same!')
    return np.sum(np.sum(((y_true - y_predict) ** 2), axis=0) / (2 * _)) / __
