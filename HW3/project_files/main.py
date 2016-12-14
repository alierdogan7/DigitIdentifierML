import csv

import math
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
                            classification_report, confusion_matrix
import sys

filepath_3and8s = "digits_3and8s.csv"


def main(args):
    dataset = convert_csv_to_dataset_3and8s(filepath_3and8s)
    predict_and_get_accuracy(dataset)


def predict_and_get_accuracy(dataset):
    # x = np.concatenate((dataset['training']['x'], dataset['validation']['x']))
    # y = np.concatenate((dataset['training']['y'], dataset['validation']['y']))
    x = dataset['training']['x']
    y = dataset['training']['y']

    epoch = 1
    theta0, theta_set = stochastic_gradient_descent(x, y, c=2, k=0.01, epoch=epoch)

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
    print("Parameters: theta0:%f, theta_set:%s, epoch:%d, max_iters:%d Accuracy: \n%f percent.\n\n" %
            (theta0, str(theta_set), epoch, accuracy))


def stochastic_gradient_descent(x, y, c=2, k=0.01, epoch=1):
    m = x.shape[0]  # number of rows
    n = x.shape[1] # number of features
    learn_rate = 1 / (c + k * epoch)

    # initial values of thetas
    theta0 = np.random.rand()
    theta_set = np.random.random(n)
    print("Initial theta0:%f, theta_set:%s" % (theta0, str(theta_set)))

    # error with initial thetas
    J = cost_function(x, y, theta0, theta_set, lamda=0.1)

    for iteration in range(epoch):
        for i in m:
            theta_gradients = []
            gradient_theta0 =  (h_theta(theta0, theta_set, x[i]) - y[i])

            for j in n: # CALCULATE THE DERIVATIVES OF EACH FEATURE OF THE CURRENT SAMPLE
                theta_gradients.append(( h_theta(theta0, theta_set, x[i]) - y[i]) * x[i][j] )

            theta0 -= learn_rate * gradient_theta0 # update theta_0
            for j in n: #update rest of thetas
                theta_set[j] -= learn_rate * theta_gradients[j] #update theta_j

            #compute the error again
            J = cost_function(x, y, theta0, theta_set, lamda=0.1)
            print("Current cost is: %f" % J)


    return theta0, theta_set


# COST FUNCTION WITH L2 REGULARIZATION
def cost_function(x, y, theta0, theta_set, lamda):
    # x is a mxn matrix for all examples' features
    # y is a mx1 matrix for all examples' classes
    m = len(x)
    n = len(theta_set)
    sum = 0

    for i in range(m):
        h_theta_x_i = h_theta(theta0, theta_set, x[i])
        sum += y[i] * math.log10(h_theta_x_i) + (1 - y[i]) * math.log10(1 - h_theta_x_i)

    cost = -sum / m

    regularization = 0
    for j in range(n):
        regularization += theta_set[j] ** 2
    regularization = regularization * lamda / (2*m)

    cost += regularization
    return cost


def h_theta(theta0, theta_set, x):
    # theta_set is a 3x1 matrixbinary classification. Pick two digits (3 and 8) and
    # create a subset of the initial MNIST data set containing only those two digits both in the training
    # set and the test set. Implement Logistic Regression
    # x is a 1x3 matrix for a single example's features
    # returns theta0 + theta1 * x1 + theta2 * x2 + theta3 * x3
    return sigmoid(theta0 + x.dot(theta_set).sum())


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def convert_csv_to_dataset_3and8s(filepath):
    dataset = {'training': {},
               'cv': {},
               'testing': {}}
    threes = []
    eights = []

    with open(filepath, 'r') as file:
        for line in csv.reader(file):
            tmp = list(map(lambda x: float(x), line[:-1]))
            if line[-1] == '3':
                threes.append(tmp) #if 3, label it 0
            elif line[-1] == '8':
                eights.append(tmp) #if 3, label it 1
            else:
                continue #just pass

    random.shuffle(threes)
    random.shuffle(eights)

    # separating into 48% / 32% / 20% , Training, CV, Testing respectively
    cv_set_offset_3s = int(len(threes) * 0.50)
    cv_set_offset_8s = int(len(eights) * 0.50)

    testing_set_offset_3s = int(len(threes) * 0.80)
    testing_set_offset_8s = int(len(eights) * 0.80)

    # Digit 3 is y=0, Digit 8 is y=1
    training_3s, cv_3s, testing_3s = threes[:cv_set_offset_3s], threes[cv_set_offset_3s:testing_set_offset_3s], \
                                     threes[testing_set_offset_3s:]
    training_8s, cv_8s, testing_8s = eights[:cv_set_offset_8s], eights[cv_set_offset_8s:testing_set_offset_8s], \
                                     eights[testing_set_offset_8s:]

    dataset['training']['x'] = np.array(training_3s)
    dataset['training']['y'] = np.array([0] * len(training_3s))
    dataset['training']['x'] = np.array(training_8s)
    dataset['training']['y'] = np.array([1] * len(training_8s))

    dataset['cv']['x'] = np.array(cv_3s)
    dataset['cv']['y'] = np.array([0] * len(cv_3s))
    dataset['cv']['x'] = np.array(cv_8s)
    dataset['cv']['y'] = np.array([1] * len(cv_8s))

    dataset['testing']['x'] = np.array(testing_3s)
    dataset['testing']['y'] = np.array([0] * len(testing_3s))
    dataset['testing']['x'] = np.array(testing_8s)
    dataset['testing']['y'] = np.array([1] * len(testing_8s))

    return dataset


def pick_3and8s():
    with open("digits.csv", 'r') as rfile, open(filepath_3and8s, 'w') as wfile:
        selected_lines = []
        writer = csv.writer(wfile)
        for line in csv.reader(rfile):
            if line[-1] == '3' or line[-1] == '8':
                selected_lines.append(line)
        writer.writerows(selected_lines)

    print("New csv file generated: %s" % filepath_3and8s)


def visualize_digit(line):
    first_digit = line[-1]
    print("Digit is: %s" % first_digit)

    converted_line = list(map(lambda x: int(float(x)*255), line))
    I = np.array(converted_line[:-1]).reshape((20, 20)).T

    plt.gray()
    plt.imsave("img/my_%s_%d.png" % (first_digit, random.randint(1,100)), I)
    # plt.imshow(I)

if __name__ == "__main__":
    main(sys.argv)