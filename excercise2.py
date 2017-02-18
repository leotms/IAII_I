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

def excercise1():

    ## 1) Data contained in x01.txt. Body weight vs. Brain weight
    ## using aplha 0.1 and 100 iterations.
    filepath   = "./data/x01.txt"
    alpha      = 0.1
    iterations = 100

    print "Using alpha 0.1 and 100 iterations."
    print "close graphics to continue..."

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

    info  = "Min cost: " + str(mincost) +"\nMax cost: " + str(maxcost) + "\nAlpha: " + str(alpha)

    plt.figtext(0.4, 0.8, info,
            bbox=dict(facecolor = 'blue', alpha=0.2),
            horizontalalignment = 'left',
            verticalalignment   = 'center')
    plt.show()

    ## Printing scatterplot vs Minimization Function

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

def excercise2():

    ## 1) Data contained in x08.txt. Mortality Rate
    ## using alpha 0.1 and 100 iterations.
    filepath   = "./data/x08.txt"
    alpha      = 0.1
    iterations = 100

    data, X, y, columns, nExamples = readData(filepath)

    print "Using alpha 0.1 and 100 iterations."
    print "close graphics to continue..."

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
    plt.title('Cost vs. Iterations\nMortality Rate: Murders per annum per 1.000.000 inhabitants')

    #Calculate min and max cost
    mincost = min(J_history)[0]
    maxcost = max(J_history)[0]

    info  = "Min cost: " + str(mincost) +"\nMax cost: " + str(maxcost) + "\nAlpha: " + str(alpha)

    plt.figtext(0.4, 0.8, info,
            bbox=dict(facecolor = 'blue', alpha=0.2),
            horizontalalignment = 'left',
            verticalalignment   = 'center')
    plt.show()

    # Now we calculate for alphas (0.1, 03, 0.5, 0.7, 0.9, 1) and 100 iterations.
    alphas = [[0.1,'g'],[0.3,'r'], [0.5,'b'], [0.7,'y'], [0.9,'c'], [1,'m']]

    fig = plt.figure()
    ax  = fig.add_subplot()

    plt.xlabel('Iteraciones')
    plt.ylabel('Cost Function')
    plt.title('Cost vs. Iterations for different learning rates\nMortality Rate: Murders per annum per 1.000.000 inhabitants')

    for alpha in alphas:

        #Add Interception data (column of ones)
        interception = np.ones(shape=(nExamples, columns))
        interception[:, 1:columns] = X

        #Init Theta and Run Gradient Descent
        theta = np.zeros(shape=(columns, 1))

        theta, J_history = gradient_descent(interception, y, theta, alpha[0], iterations)
        #this plots the Coast Function vs the number of Iterations
        mincost = min(J_history)[0]
        maxcost = max(J_history)[0]
        info    = "\nAlpha: " + str(alpha[0]) + ", mincost: " + str(mincost) +", maxcost: " + str(maxcost)

        p1 = plt.plot(np.arange(iterations), J_history, c=alpha[1], label = info)

    plt.legend(loc=1)
    plt.show()

if __name__ =="__main__":
    '''
        Main function.
    '''

    print "Starting Exc. 1..."
    print "Body weight vs. Brain weight"
    excercise1()
    print "done."
    print "Starting Exc. 2..."
    print "Mortality Rate: Murders per annum per 1.000.000 inhabitants"
    excercise2()
    print "done."
