'''
    File:        excercise2.py
    Description: Contains code to calculate project's second activity
    Authors:     Joel Rivas        #11-10866
                 Leonardo Martinez #11-10576
                 Nicolas Manan     #06-39883
    Updated:     02/12/2017
'''

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import plot, show, xlabel, ylabel
from gradient_descent import *

def parse_args(args):

    ''' Parses the command line data arguments:
            needed:
                1 = File path
                2 = Number of iterations
                3 = Learning rate alpha
            optional flags:
                -p: displays scatterplot of the data alongside the function.
                -n: normalize the data
                If no flags are passed default behavior is set to false.
    '''


    if len(args) < 4:
        raise Exception("ERROR: Missing arguments.")

    filepath   = args[1]
    iterations = int(args[2])
    alpha      = float(args[3])

    normalize  = False
    plot       = False

    if len(args) == 5:
        if args[4] == '-n':
            normalize = True
        elif args[4] == '-p':
            plot = True
    if len(args) == 6:
        if args[5] == '-n':
            normalize = True
        elif args[5] == '-p':
            plot = True

    return filepath, iterations, alpha, normalize, plot

if __name__ =="__main__":

    '''
        Main function.
    '''

    filepath, iterations, alpha, isnormalize, isplot = parse_args(sys.argv)

    print iterations

    # Reading the data and extracting the information from the columns
    data = np.loadtxt(filepath, delimiter=',')

    columns = data.shape[1]

    X = data[:, :columns - 1] #stores features columns
    y = data[:,  columns - 1] #stores objective

    nExamples = y.size

    y.shape = (nExamples, 1)

    #Normalization of the features
    if isnormalize:
        X, mean_r, std_r = normalize(X)

    #Add Interception data (column of ones)
    interception = np.ones(shape=(nExamples, columns))
    interception[:, 1:columns] = X

    #Init Theta and Run Gradient Descent
    theta = np.zeros(shape=(columns, 1))

    theta, J_history = gradient_descent(interception, y, theta, alpha, iterations)

    #this plots the Coast Function vs the number of Iterations
    plot(np.arange(iterations), J_history)
    xlabel('Iterations')
    ylabel('Cost Function')
    show()

    if (isplot):

        if columns > 3 :
            print "This script doens't plot scatter vs funtion for more than two variables."
        if columns == 2:

            fig = plt.figure()
            ax = fig.add_subplot()
            xs = data[:, 0]
            ys = data[:, 1]
            zs = []

            #calculate z values according to theta
            for value in xs:
                zs.append(theta[0][0] + theta[1][0]*value)

            plt.scatter(xs, ys, c='b', marker='o')
            plt.plot(xs, zs, c='r')

            plt.show()

        if columns == 3:

            #Plot the data

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            xs = data[:, 0]
            ys = data[:, 1]
            zs = data[:, 2]
            qs = []

            #calculate q values according to theta
            for num in range(len(xs)):
                qs.append(theta[0][0] + theta[1][0]*(xs[num]) + theta[2][0]*(ys[num]))

            print len(xs), len(ys), len(zs), len(qs)
            ax.scatter(xs, ys, zs, c='b', marker='o')
            ax.plot(xs,ys,qs, c='r')

            plt.show()
