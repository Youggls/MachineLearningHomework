import numpy as np
from util import *
from plot import *
import scipy.optimize as opt

input_size = 400
hidden_size = 25
num_labels = 10

mat_data = read_matlab('./ex4/ex4data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
m, n = X.shape

data_trained = read_matlab('./ex4/ex4weights.mat')
trained_theta1 = data_trained['Theta1']
trained_theta2 = data_trained['Theta2']
theta_pretrained = np.hstack((trained_theta1.flatten(), trained_theta2.flatten()))
print('The cost function without regularization is {}.'.format(cost(theta_pretrained, input_size, hidden_size, num_labels, X, y, 0)[0]))
print('The cost function with regularization is {}.'.format(cost(theta_pretrained, input_size, hidden_size, num_labels, X, y, 1)[0]))
theta1 = rand_initialize_weights(input_size, hidden_size)
theta2 = rand_initialize_weights(hidden_size, num_labels)
theta = np.hstack((theta1.ravel(), theta2.ravel()))

l = 1.0
result = opt.minimize(fun=cost, x0=theta,
                      args=(input_size, hidden_size, num_labels, X, y, l),
                      method='TNC', jac=True, options={'maxiter': 150})
theta_trained = result.x
theta1 = np.reshape(theta_trained[0:(hidden_size * (input_size + 1)), ],
                             (hidden_size, input_size + 1))
theta2 = np.reshape(theta_trained[(hidden_size * (input_size + 1)):, ],
                             (num_labels, hidden_size + 1))

pred = predict(theta1, theta2, X)
print ('\033[1;32mAccuracy is:{}%.\033[0m'.format(np.mean(pred == y) * 100))
show_100img(X)
