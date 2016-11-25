import csv
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn import svm


filepath_3and8s = "digits_3and8s.csv"


def main():
    # pick_3and8s()
    # with open(new_filename, 'r') as file:
    #     lines = list(csv.reader(file))
    #     for line in lines[:10]:
    #         visualize_digit(line)

    dataset = convert_csv_to_dataset(filepath_3and8s)
    linear_svm(dataset)

def linear_svm(dataset):
    lin_svm = svm.LinearSVC()
    lin_svm.fit(dataset['training']['x'], dataset['training']['y'])
    print(lin_svm.score(dataset['cv']['x'], dataset['cv']['y']))


def convert_csv_to_dataset(filepath):
    dataset = {'training': {'x': [], 'y': []}, 'cv': {'x': [], 'y': []}}
    threes = []
    eights = []

    with open(filepath, 'r') as file:
        for line in csv.reader(file):
            tmp = list(map(lambda x: float(x), line[:-1]))
            if line[-1] == '3':
                threes.append(tmp)
            elif line[-1] == '8':
                eights.append(tmp)
            else:
                continue #just pass

    random.shuffle(threes)
    random.shuffle(eights)

    # separating into 80% / 20% , Training vs. CV respectively
    cv_set_offset_3s = int(len(threes) * 0.8)
    cv_set_offset_8s = int(len(eights) * 0.8)

    # Digit 3 is y=0, Digit 8 is y=1
    dataset['training']['x'] += threes[:cv_set_offset_3s]
    dataset['training']['y'] += [0] * cv_set_offset_3s #dont break this order!!!

    dataset['training']['x'] += eights[:cv_set_offset_8s]
    dataset['training']['y'] += [1] * cv_set_offset_8s

    dataset['cv']['x'] += threes[cv_set_offset_3s:]
    dataset['cv']['y'] += [0] * cv_set_offset_3s #dont break this order!!!

    dataset['cv']['x'] += eights[cv_set_offset_8s:]
    dataset['cv']['y'] += [1] * cv_set_offset_8s

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