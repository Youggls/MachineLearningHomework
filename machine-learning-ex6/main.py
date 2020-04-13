from util import *
from plot import *
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data1 = read_matlab('./ex6/ex6data1.mat')
    X1 = data1['X']
    y1 = data1['y'].ravel()
    data2 = read_matlab('./ex6/ex6data2.mat')
    X2 = data2['X']
    y2 = data2['y'].ravel()
    data3 = read_matlab('./ex6/ex6data3.mat')
    X3 = data3['X']
    y3 = data3['y'].ravel()
    X3_val = data3['Xval']
    y3_val = data3['yval'].ravel()
    svm1 = svm.LinearSVC(C=1)
    svm1.fit(X1, y1)
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)
    print('Self gaussian kernel function is {}.'.format(sim))
    svm2 = svm.SVC(C=100, kernel='rbf', gamma=10)
    svm2.fit(X=X2, y=y2)
    C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    gamma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_score = -1
    best_C = -1
    best_gamma = -1
    best_SVM = None
    for C in C_list:
        for gamma in gamma_list:
            svm3 = svm.SVC(C=C, gamma=gamma, kernel='rbf')
            svm3.fit(X3, y3)
            score = svm3.score(X3_val, y3_val)
            if score > max_score:
                max_score = score
                best_C = C
                best_gamma = gamma
                best_SVM = svm3
    print('The best gamma value is {}, best C is {}.'.format(best_gamma, best_C))
    plot(X1, y1, svm1, X2, y2, svm2, X3, y3, best_SVM)
