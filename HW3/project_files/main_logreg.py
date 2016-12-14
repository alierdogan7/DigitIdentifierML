import pprint

from math import log10
import sys
import math
import numpy as np

def main(argv):
    filename = argv[0]
    normalized_data = read_csv(filename)
    predict_and_get_accuracy(normalized_data)


def predict_and_get_accuracy(normalized_data):
    dataset = read_csv(normalized_data)
    x = np.concatenate((dataset['training']['x'] ,dataset['validation']['x']))
    y = np.concatenate((dataset['training']['y'] , dataset['validation']['y']))
    # x = dataset['training']['x']
    # y = dataset['training']['y']

    alpha = 0.1 # optimized 0.1
    epsilon = 0.00001    # optimized: 0.00001
    max_iters = 3000
    theta0, theta_set = gradient_descent(x, y, alpha, max_iters, epsilon)

    #now we've learned our parameters. let's get accuracy with cross-validation set
    x = dataset['test']['x']
    y = dataset['test']['y']
    m = len(x)

    correct_count = 0
    for i in range(m):
        result = h_theta(theta0, theta_set, x[i])
        if result >= 0.5:
            predict = 1
            if predict == y[i]:
                correct_count += 1
        else:
            predict = 0
            if predict == y[i]:
                correct_count += 1

    accuracy = correct_count / float(m) * 100

    '''with open('log.txt', "a") as logfile:
        logfile.write("Accuracy with theta0:%f, theta_set:%s, alpha:%f, epsilon:%f, max_iters:%d: %f percent.\n\n" %
                            (theta0, str(theta_set), alpha, epsilon, max_iters, accuracy))'''

    print("Trained on Training+Cross Validation Set\nAccuracy is tested on testing set\n")
    print("Parameters: theta0:%f, theta_set:%s, alpha:%f, epsilon:%f, max_iters:%d Accuracy: \n%f percent.\n\n" % 
            (theta0, str(theta_set), alpha, epsilon, max_iters, accuracy))


def gradient_descent(x, y, alpha=0.001, max_iterations=1000, ep=0.00001):
    converged = False
    m = x.shape[0]  # number of rows
    iterations = 0

    # initial values of thetas
    theta0 = np.random.rand()
    theta_set = np.random.random(x.shape[1])
    print("Initial theta0:%f, theta_set:%s" % (theta0, str(theta_set)))
    # 1.089358, theta_set:[-0.10824952 -1.04552378  0.69685797]
    # theta0 = 1.089358
    # theta_set = [-0.10824952, -1.04552378, 0.69685797]

    # error with initial thetas
    J = cost_function(x, y, theta0, theta_set)

    while not converged:
        grad_theta0 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) for i in range(m)])
        grad_theta1 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) * x[i][0] for i in range(m)])
        grad_theta2 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) * x[i][1] for i in range(m)])
        grad_theta3 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) * x[i][2] for i in range(m)])

        #update thetas
        theta0 -= alpha * grad_theta0
        theta_set[0] -= alpha * grad_theta1 #update theta1
        theta_set[1] -= alpha * grad_theta2 #update theta2
        theta_set[2] -= alpha * grad_theta3 #update theta3

        #compute the error again
        e = cost_function(x, y, theta0, theta_set)
        print("Current cost is: %f" % e)

        if abs(J-e) <= ep:
            converged = True
            print("Converged with %d iterations." % iterations)

        J = e #updating the cost with new thetas
        iterations += 1

        if iterations >= max_iterations:
            converged = True
            print("Reached max. iterations: %d ." % iterations)

    return theta0, theta_set


def cost_function(x, y, theta0, theta_set):
    # x is a mx3 matrix for all examples' features
    # y is a mx1 matrix for all examples' classes
    m = len(x)
    sum = 0

    for i in range(m):
        h_theta_x_i = h_theta(theta0, theta_set, x[i])
        sum += y[i] * log10(h_theta_x_i) + (1 - y[i]) * log10(1 - h_theta_x_i)

    return -sum / m


def h_theta(theta0, theta_set, x):
    # theta_set is a 3x1 matrix
    # x is a 1x3 matrix for a single example's features
    # returns theta0 + theta1 * x1 + theta2 * x2 + theta3 * x3
    return sigmoid(theta0 + x.dot(theta_set).sum())


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# X is a data set matrix with parameters: x1 Pclass, x2 Sex, x3 Age
def read_csv(filename):
    splitted_data = {'training': {'x': [], 'y': []},
                     'validation': {'x': [], 'y': []},
                     'test': {'x': [], 'y': []}}

    dataset = np.loadtxt(open("titanicdata.csv", "rb"), delimiter=",", skiprows=1)
    x = dataset[:, [1, 2, 3]] # separating features and classes
    y = dataset[:, [0]]

    normalize_features(x)

    splitted_data['training']['x'] = x[:400]
    splitted_data['training']['y'] = y[:400]
    splitted_data['validation']['x'] = x[400:700]
    splitted_data['validation']['y'] = y[400:700]
    splitted_data['test']['x'] = x[700:]
    splitted_data['test']['y'] = y[700:]

    return splitted_data


def normalize_features(x):
    age_set = [data[2] for data in x]  # index 2 corresponds to age column
    min_age, max_age = min(age_set), max(age_set)

    for index in range(len(x)):
        x[index][2] = (x[index][2] - min_age) / (max_age - min_age)


if __name__ == "__main__":
    main(sys.argv[1:])
