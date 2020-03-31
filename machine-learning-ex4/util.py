import scipy.io as scio
import numpy as np

def read_matlab(path):
    return scio.loadmat(path)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(y_pred):
    return np.multiply(sigmoid(y_pred), sigmoid(1 - y_pred))

def trans_label_to_onehot(label):
    length, _ = label.shape
    res = np.zeros((length, 10))
    for i in range(length):
        res[i, label[i,0] - 1] = 1
    return res


def cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, l):
    Theta_1 = np.reshape(nn_params[0:(hidden_layer_size * (input_layer_size + 1)), ],
                         (hidden_layer_size, input_layer_size + 1))
    Theta_2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):, ],
                         (num_labels, hidden_layer_size + 1))

    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))

    Z_2 = X.dot(Theta_1.T)
    A_2 = sigmoid(Z_2)
    A_2 = np.hstack((np.ones((m, 1)), A_2))

    Z_3 = A_2.dot(Theta_2.T)
    A_3 = sigmoid(Z_3)

    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] - 1] = 1

    j = 0.0
    for i in range(m):
        j += np.log(A_3[i, ]).dot(-Y[i, ].T) - np.log(1 - A_3[i, ]).dot(1 - Y[i, ].T)
    j /= m

    Theta_1_square = np.square(Theta_1[:, 1:])
    Theta_2_square = np.square(Theta_2[:, 1:])
    reg = 1.0 * l / (2 * m) * (np.sum(Theta_1_square) + np.sum(Theta_2_square))
    j += reg

    d_3 = A_3 - Y
    D_2 = d_3.T.dot(A_2)

    Z_2 = np.hstack((np.ones((m, 1)), Z_2))
    d_2 = d_3.dot(Theta_2) * sigmoid_gradient(Z_2)
    d_2 = d_2[:, 1:]
    D_1 = d_2.T.dot(X)

    Theta_1_grad = 1.0 * D_1 / m
    Theta_1_grad[:, 1:] = Theta_1_grad[:, 1:] + 1.0 * l / m * Theta_1[:, 1:]

    Theta_2_grad = 1.0 * D_2 / m
    Theta_2_grad[:, 1:] = Theta_2_grad[:, 1:] + 1.0 * l / m * Theta_2[:, 1:]

    grad = np.hstack((Theta_1_grad.ravel(), Theta_2_grad.ravel()))

    return j, grad

def calc_accuracy(y_true, y_pred):
    """
    y_true and y_pred must be n * 1
    """
    total, _ = y_true.shape
    TP = 0
    for i in range(total):
        if y_true[i, 0] == y_pred[i, 0]:
            TP += 1
    return TP/total * 100

def rand_initialize_weights(l_in, l_out):
    epsilon_init = 0.12
    theta = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return theta

def predict(theta1, theta2, X):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    a2 = sigmoid(X.dot(theta1.T))
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = sigmoid(a2.dot(theta2.T))

    p = np.argmax(a3, axis=1)
    p += 1  # The theta1 and theta2 are loaded from Matlab data, in which the matrix index starts from 1.
    return p
