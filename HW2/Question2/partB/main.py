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


def main(args):
    dataset = convert_csv_to_dataset_3and8s(filepath_3and8s)
    poly_svm_training(dataset)
    poly_svm_testing(dataset, best_C=100, best_degree=5)


# Best C is 100 and best degree is 5 for my case
def poly_svm_training(dataset):
    for tmp_C in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000000]:
        print()
        for tmp_degree in [2, 3, 4, 5, 6]:
            # lin_svm = svm.LinearSVC(C=C)
            polysvm = svm.SVC(C=tmp_C, kernel='poly', degree=tmp_degree, coef0=1)

            # WE WILL PERFORM TWO-FOLD CROSS-VALIDATION.. BEHOLD!!
            polysvm.fit(dataset['training']['x'], dataset['training']['y'])
            first_score = polysvm.score(dataset['cv']['x'], dataset['cv']['y'])*100

            # WE ARE SWAPPING TRAINING AND CV PORTIONS FOR PERFORMING TWO-FOLD C.V.
            tmp_x = dataset['cv']['x']
            tmp_y = dataset['cv']['y']
            dataset['cv']['x'] = dataset['training']['x']
            dataset['cv']['y'] = dataset['training']['y']
            dataset['training']['x'] = tmp_x
            dataset['training']['y'] = tmp_y

            polysvm.fit(dataset['training']['x'], dataset['training']['y'])
            second_score = polysvm.score(dataset['cv']['x'], dataset['cv']['y'])*100

            print("(Poly) C: %f, degree: %d  == %f" % (tmp_C, tmp_degree, (first_score + second_score) / 2))


def poly_svm_testing(dataset, best_C, best_degree):
        polysvm = svm.SVC(C=best_C, kernel='poly', degree=best_degree, coef0=1)
        polysvm.fit(dataset['training']['x']+dataset['cv']['x'], dataset['training']['y']+dataset['cv']['y'])
        predictions = polysvm.predict(dataset['testing']['x'])
        real_y_vals = dataset['testing']['y']

        print("\n\nScores using C = %f, degree = %d" % (best_C, best_degree))

        print('Accuracy:', accuracy_score(real_y_vals, predictions))
        print('F1 score:', f1_score(real_y_vals, predictions,average='weighted'))
        print('Recall:', recall_score(real_y_vals, predictions,
                                      average='weighted'))
        print('Precision:', precision_score(real_y_vals, predictions,
                                            average='weighted'))
        print('\n clasification report:\n', classification_report(real_y_vals, predictions))
        print('\n confussion matrix:\n',confusion_matrix(real_y_vals, predictions))


def convert_csv_to_dataset_3and8s(filepath):
    dataset = {'training': {'x': [], 'y': []},
               'cv': {'x': [], 'y': []},
               'testing': {'x': [], 'y': []}}
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

    dataset['training']['x'] += training_3s
    dataset['training']['y'] += [0] * len(training_3s)
    dataset['training']['x'] += training_8s
    dataset['training']['y'] += [1] * len(training_8s)

    dataset['cv']['x'] += cv_3s
    dataset['cv']['y'] += [0] * len(cv_3s)
    dataset['cv']['x'] += cv_8s
    dataset['cv']['y'] += [1] * len(cv_8s)

    dataset['testing']['x'] += testing_3s
    dataset['testing']['y'] += [0] * len(testing_3s)
    dataset['testing']['x'] += testing_8s
    dataset['testing']['y'] += [1] * len(testing_8s)

    return dataset

if __name__ == "__main__":
    main(sys.argv[1:])