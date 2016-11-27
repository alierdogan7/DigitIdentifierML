import csv
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
                            classification_report, confusion_matrix

filepath_3and8s = "digits_3and8s.csv"


def main():
    # pick_3and8s()
    # with open(new_filename, 'r') as file:
    #     lines = list(csv.reader(file))
    #     for line in lines[:10]:
    #         visualize_digit(line)

    dataset = convert_csv_to_dataset(filepath_3and8s)
    # linear_svm(dataset)
    poly_svm(dataset)


def linear_svm(dataset):
    # for C in [0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.01, 0.1, 1, 10, 100, 1000, ]:
    #     # lin_svm = svm.LinearSVC(C=C)
    #     lin_svm = svm.SVC(C=C, kernel='linear')
    #
    #     # WE WILL PERFORM TWO-FOLD CROSS-VALIDATION.. BEHOLD!!
    #     lin_svm.fit(dataset['training']['x'], dataset['training']['y'])
    #     first_score = lin_svm.score(dataset['cv']['x'], dataset['cv']['y'])*100
    #
    #     # WE ARE SWAPPING TRAINING AND CV PORTIONS FOR PERFORMING TWO-FOLD C.V.
    #     tmp_x = dataset['cv']['x']
    #     tmp_y = dataset['cv']['y']
    #     dataset['cv']['x'] = dataset['training']['x']
    #     dataset['cv']['y'] = dataset['training']['y']
    #     dataset['training']['x'] = tmp_x
    #     dataset['training']['y'] = tmp_y
    #
    #     lin_svm.fit(dataset['training']['x'], dataset['training']['y'])
    #     second_score = lin_svm.score(dataset['cv']['x'], dataset['cv']['y'])*100
    #
    #     print("(Linear) C: %f == %f" % (C, (first_score + second_score) / 2.0))

    # lin_svm = svm.LinearSVC(C=0.01)
    lin_svm = svm.SVC(C=0.01, kernel='linear')
    lin_svm.fit(dataset['training']['x']+dataset['cv']['x'], dataset['training']['y']+dataset['cv']['y'])
    predictions = lin_svm.predict(dataset['testing']['x'])
    real_y_vals = dataset['testing']['y']

    print('Accuracy:', accuracy_score(real_y_vals, predictions))
    print('F1 score:', f1_score(real_y_vals, predictions,average='weighted'))
    print('Recall:', recall_score(real_y_vals, predictions,
                                  average='weighted'))
    print('Precision:', precision_score(real_y_vals, predictions,
                                        average='weighted'))
    print('\n clasification report:\n', classification_report(real_y_vals, predictions))
    print('\n confussion matrix:\n',confusion_matrix(real_y_vals, predictions))


def poly_svm(dataset):
    for C in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000000]:
        for degree in [2, 3, 4, 5, 6]:
            # lin_svm = svm.LinearSVC(C=C)
            polysvm = svm.SVC(C=C, kernel='poly', degree=degree, coef0=1)

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

            print("(Poly) C: %f, degree: %d  == %f" % (C, degree, (first_score + second_score) / 2))

    # lin_svm = svm.LinearSVC(C=0.01)
    # lin_svm.fit(dataset['training']['x']+dataset['cv']['x'], dataset['training']['y']+dataset['cv']['y'])
    # predictions = lin_svm.predict(dataset['testing']['x'])
    # real_y_vals = dataset['testing']['y']
    #
    # print('Accuracy:', accuracy_score(real_y_vals, predictions))
    # print('F1 score:', f1_score(real_y_vals, predictions,average='weighted'))
    # print('Recall:', recall_score(real_y_vals, predictions,
    #                               average='weighted'))
    # print('Precision:', precision_score(real_y_vals, predictions,
    #                                     average='weighted'))
    # print('\n clasification report:\n', classification_report(real_y_vals, predictions))
    # print('\n confussion matrix:\n',confusion_matrix(real_y_vals, predictions))

def convert_csv_to_dataset(filepath):
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

main()