'''
    File:        gradient_descent.py
    Description: This file contains the definitios to calculate gradient descent
                 for multiple linear regretion.
    Authors:     Joel Rivas        #11-10866
                 Leonardo Martinez #11-10576
                 Nicolas Manan     #06-39883
    Updated:     02/12/2017
'''

import numpy as np


def normalize(X):
    '''
        Normalizes the data provided to the gradient descent function.
    '''
    vector_mean = []
    vector_std  = []

    #this is the normalized version of X
    XN = X

    n_c = X.shape[1]
    for i in range(n_c):
        m = np.mean(X[:, i])
        s = np.std(X[:, i])
        vector_mean.append(m)
        vector_std.append(s)
        XN[:, i] = (XN[:, i] - m) / s

    return XN, vector_mean, vector_std


def cost(X, y, theta):
    '''
    Computes cost for linear regression
    '''
    #Number of training samples
    m = y.size

    pred = X.dot(theta)

    sqErrors = (pred - y)

    cost = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        pred = X.dot(theta)

        theta_size = theta.size

        for j in range(theta_size):

            temp = X[:, j]
            temp.shape = (m, 1)

            errors_x1 = (pred - y) * temp

            theta[j][0] = theta[j][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = cost(X, y, theta)

    return theta, J_history
