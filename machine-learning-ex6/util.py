import numpy as np
import scipy.io as scio

def read_matlab(path):
    return scio.loadmat(path)

def gaussian_kernel(x1, x2, sigma=10):
    return np.exp(-np.sum(np.square(x1 - x2)) / (2.0 * sigma ** 2))
