'''
    File:        excercise2.py
    Description: Contains code to calculate project's second excercise
    Authors:     Joel Rivas        #11-10866
                 Leonardo Martinez #11-10576
                 Nicolas Manan     #06-39883
    Updated:     02/12/2017
'''

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient_descent import *

def readData(filepath):

    # Reading the data and extracting the information from the columns
    data = np.loadtxt(filepath, delimiter=',')

    columns = data.shape[1]

    X = data[:, :columns - 1] #stores features columns
    y = data[:,  columns - 1] #stores objective

    nExamples = y.size

    y.shape = (nExamples, 1)

    return data, X, y, columns, nExamples

if __name__ =="__main__":
    '''
        Main function.
    '''
    ## 1) Data contained in x01.txt. Body weight vs. Brain weight
    ## using aplha 0.1 and 100 iterations.
    filepath   = "./data/x01.txt"
    alpha      = 0.1
    iterations = 100

    data, X, y, columns, nExamples = readData(filepath)

    #Normalization of the features
    X, mean_r, std_r = normalize(X)

    #Add Interception data (column of ones)
    interception = np.ones(shape=(nExamples, columns))
    interception[:, 1:columns] = X

    #Init Theta and Run Gradient Descent
    theta = np.zeros(shape=(columns, 1))

    theta, J_history = gradient_descent(interception, y, theta, alpha, iterations)

    #this plots the Coast Function vs the number of Iterations
    p1 = plt.plot(np.arange(iterations), J_history)
    #Labels
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.title('Cost vs. Iterations\nBrain Weight vs. Body Weight')

    #Calculate min and max cost
    mincost = min(J_history)[0]
    maxcost = max(J_history)[0]

    info  = "Costo minimo: " + str(mincost) +"\nCosto maximo: " + str(maxcost) + "\nAlpha: " + str(alpha)

    plt.figtext(0.4, 0.8, info,
            bbox=dict(facecolor = 'blue', alpha=0.2),
            horizontalalignment = 'left',
            verticalalignment   = 'center')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot()
    xs = data[:, 0]
    ys = data[:, 1]
    zs = []

    plt.xlabel('Brain Weight')
    plt.ylabel('Body Weight')
    plt.title('Scatterplot vs. Minimization Function\nBrain Weight vs. Body Weight')

    #calculate z values according to theta
    for value in xs:
        zs.append(theta[0][0] + theta[1][0]*value)

    p1 = plt.scatter(xs, ys, c='b', marker='o', label = "Scatter Plot")
    p2 = plt.plot(xs, zs, c='r', label = "M. Function")
    plt.legend(loc=2)

    plt.show()
