import csv
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
                            classification_report, confusion_matrix
import sys

filepath_3and8s = "digits_3and8s.csv"
filepath = "digits.csv"
filepath_first_4000 = "digits_first_4000.csv"
filepath_last_1000 = "digits_last_1000.csv"


def main(args):
    ##################
    ## PART C
    ##################

    classifiers = []

    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        #e.g. element 0 corresponds to digit 0's classifier
        print("Training classifier for digit %s ..." % digit)
        classifiers.append(part_c_generate_classifier(digit))

    testing_dataset = []
    testing_real_labels = []
    testing_predicted_labels = []

    with open(filepath_last_1000, 'r') as file:
        print("Reading 1000 samples for testing ...")
        for line in csv.reader(file):
            testing_dataset.append(list(map(lambda x: float(x), line[:-1])))
            testing_real_labels.append(line[-1])

    print("Obtaining probabilistic predictions of all 10 classifiers for each sample...")
    i = 0
    for features, label in zip(testing_dataset, testing_real_labels):
        i += 1
        print("Sample %d's predictions being calculated..." % i)
        predicts_for_sample = []
        for classifier in classifiers:
            predicts_for_sample.append(list(zip(classifier.classes_, classifier.predict_proba(np.array(features).reshape(1, -1))[0]))[0][1])

        # choose the index of the classifier which is the most probable for this sample
        max_index = 0
        for index, val in enumerate(predicts_for_sample):
            if predicts_for_sample[index] >= predicts_for_sample[max_index]:
                max_index = index

        testing_predicted_labels.append(max_index)

    print("Computing accuracy now...")
    correct = 0
    for predicted, real in zip(testing_predicted_labels, testing_real_labels):
        if predicted == int(real):
            correct += 1

    print("%d correct predictions over %d samples" % (correct, len(testing_real_labels)))
    print("Overall accuracy is %f" % (float(correct) * 100 / len(testing_real_labels)))

    testing_real_labels = list(map(lambda x: int(x), testing_real_labels))
    print("\n\n--------------\n\nConfusion matrix\n\n", confusion_matrix(testing_real_labels, testing_predicted_labels))

    print('Accuracy:', accuracy_score(testing_real_labels, testing_predicted_labels))
    print('F1 score:', f1_score(testing_real_labels, testing_predicted_labels, average='weighted'))
    print('Recall:', recall_score(testing_real_labels, testing_predicted_labels,
                                  average='weighted'))
    print('Precision:', precision_score(testing_real_labels, testing_predicted_labels,
                                        average='weighted'))
    print('\n clasification report:\n', classification_report(testing_real_labels, testing_predicted_labels))


def part_c_generate_classifier(digit):
    dataset = convert_csv_to_dataset_partC(digit)

    polysvm = svm.SVC(C=100, kernel='poly', degree=5, coef0=1, probability=True)
    polysvm.fit(dataset['training']['x']+dataset['cv']['x'], dataset['training']['y']+dataset['cv']['y'])

    return polysvm


def convert_csv_to_dataset_partC(digit):
    dataset = {'training': {'x': [], 'y': []},
               'cv': {'x': [], 'y': []}}

    digits = []
    others = []

    with open(filepath_first_4000, 'r') as file:
        for line in csv.reader(file):
            tmp = list(map(lambda x: float(x), line[:-1]))
            if line[-1] == digit:
                digits.append(tmp)
            else:
                others.append(tmp)

    random.shuffle(digits)
    random.shuffle(others)

    # separating into 50% / 50% , Training, CV, respectively
    cv_set_offset_digits = int(len(digits) * 0.50)
    cv_set_offset_others = int(len(others) * 0.50)

    # Digit 'digit' is y=1, else y=0
    training_digits, cv_digits = digits[:cv_set_offset_digits], digits[cv_set_offset_digits:]
    training_others, cv_others = others[:cv_set_offset_others], others[cv_set_offset_others:]

    dataset['training']['x'] += training_digits
    dataset['training']['y'] += [0] * len(training_digits)
    dataset['training']['x'] += training_others
    dataset['training']['y'] += [1] * len(training_others)

    dataset['cv']['x'] += cv_digits
    dataset['cv']['y'] += [0] * len(cv_digits)
    dataset['cv']['x'] += cv_others
    dataset['cv']['y'] += [1] * len(cv_others)

    return dataset



if __name__ == "__main__":
    main(sys.argv[1:])