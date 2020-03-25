import numpy as np
from util import read_matlab, trans_label_to_onehot, calc_precision
from plot import show_img, show_100img
from lr import LogisticRegression, load_lr
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    # Read data from mat file.
    data = read_matlab('./ex3/ex3data1.mat')
    X = data['X']
    label = data['y']
    # Trans the label to onehot
    label_onehot = trans_label_to_onehot(label).transpose()
    # Show img
    show_100img(X)
    lr = LogisticRegression(400, 10)
    # This method uses the lib function, if you want to use self create train method, use function LogisticRegression.train
    lr.lib_train(X, label_onehot, 1)

    # You can use save and load function to save model file
    # lr.save_model('./model/lr.model')
    # lr = load_lr('./model/lr.model')
    ans = lr.predict_class(X.transpose())
    print('\033[1;32mThe accuracy of Logistsic regression is {}%.\033[0m'.format(calc_precision(label, ans)))

    # Neural network
    nn_weight = read_matlab('./ex3/ex3weights.mat')
    theta1 = nn_weight['Theta1']
    theta2 = nn_weight['Theta2']
    nn = NeuralNetwork(400, 25, 10)
    nn.set_weight(theta1, theta2)
    y_pred = nn.predict_class(X)
    y_pred.shape = (5000, 1)
    print('\033[1;32mThe accuracy of neural work is {}%.\033[0m'.format(calc_precision(label, y_pred)))

