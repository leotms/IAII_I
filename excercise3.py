import sys
from gradient_descent import *

def readData(filepath):

    # Reading the data and extracting the information from the columns
    data = np.loadtxt(filepath, delimiter=',', usecols=range(3,81))

    columns = data.shape[1]

    X = data[:, :columns - 1] #stores features columns
    y = data[:,  columns - 1] #stores objective

    nExamples = y.size

    y.shape = (nExamples, 1)

    return data, X, y, columns, nExamples

def train(filepath):
    'Returns the model obtained from using Gradient Descent over a training sample'

    ## using alpha 0.7 and 100 iterations.
    alpha      = 0.7
    iterations = 1000

    data, X, y, columns, nExamples = readData(filepath)

    #Normalization of the features
    X, mean, std = normalize(X)

    print "Training model using Alpha: " + str(alpha) + ' with ' + str(iterations)
    print "N.Examples: " + str(nExamples)

    #Add Interception data (column of ones)
    interception = np.ones(shape=(nExamples, columns))
    interception[:, 1:columns] = X

    #Init Theta and Run Gradient Descent
    theta = np.zeros(shape=(columns, 1))

    theta, J_history = gradient_descent(interception, y, theta, alpha, iterations)


    return theta, mean, std

def test(model, mean, std, filepath):
    '''
        Tests every vector contained in filepath with a given model.
    '''

    data, X, y, columns, nTest = readData(filepath)

    print "Testing model using " + str(nTest) + ' vectors.'

    nRows =  data.shape[0]
    predictions = []
    for row in range(nRows):
        vector = [1.0] #
        for i in range(len(X[row])):
            vector.append((X[row][i] - mean[i])/std[i])
        predicted = np.array(vector).dot(model)
        predictions.append(predicted)
        print row, '\t', X[row], '\t' ,predicted, '\t', y[row]

    # Metrics
    # Bias: mean (predicted - y)
    subtract = np.subtract(predicted, y)
    bias = np.mean(subtract)
    # Maximun Deviation: max |y - predicted|
    subtract = np.subtract(y, predicted)
    absolute = np.absolute(subtract)
    md       = np.max(absolute)
    # Mean Absolute Deviation: mean |y - predicted|
    mad      = np.mean(absolute)
    # Mean Square Error: mean (y - predicted)^2
    square   = np.power(absolute,2)
    msq      = np.mean(square)

    print "Metrics"
    print "Bias: ", bias
    print "Maximun Deviation: ", md
    print "Mean Absolute Deviation: ", mad
    print "Mean Square Error: ", msq

if __name__ =="__main__":
    '''
        Main function.
    '''

    if len(sys.argv) < 3:
        raise Exception("ERROR: Missing arguments.")

    trainset = sys.argv[1]
    testset  = sys.argv[2]

    model, mean, std = train(trainset)

    print "Model found: ", model

    test(model, mean, std, testset)
