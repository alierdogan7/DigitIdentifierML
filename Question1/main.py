import csv
import random
import matplotlib.pyplot as plt
import numpy as np

filepath_first_4000 = "digits_first_4000.csv"


def main():
    dataset = convert_csv_to_dataset()
    samples = np.array(dataset['training']['x'])
    labels = np.array(dataset['training']['y'])

    mean_vector = [] #index i gives the mean of i-th feature
    for feature_col in range(400): #for each feature
        mean_vector.append(np.mean(samples[:, feature_col]))

    mean_vector = np.array(mean_vector)
    samples = samples - mean_vector
    cov_mat = np.cov([ samples[:, i] for i in range(400)])

    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    for i in range(len(eig_val_cov)):
        eigvec_cov = eig_vec_cov[:,i].reshape(1,400).T

        # print('Eigenvector {}: \n{}'.format(i+1, eigvec_cov))
        # print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))

    # (eigenval, eigenvec)
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    eig_pairs.sort(key = lambda x: x[0], reverse=True)

    # PART B
    # plot_eigenvalues(eig_pairs)

    # PART C
    # top_5 = map(lambda pair: pair[1], eig_pairs[:5])
    # bottom_5 = map(lambda pair: pair[1], eig_pairs[-5:])
    #
    # for key, eigvec in enumerate(top_5):
    #     visualize_digit(eigvec, "eigvec_top_%d" % (key + 1))
    #
    # for key, eigvec in enumerate(bottom_5):
    #     visualize_digit(eigvec, "eigvec_bottom_%d" % (5 - key))

    # PART D
    # matrix_U = np.array([ pair[1] for pair in eig_pairs[:2]]) #we choose first k=2 cols of eigenvector matrix
    # transformed = matrix_U.dot(samples.T).T
    # # assert transformed.shape == (400,2), "The matrix is not 400x2 dimensional."
    # plot_part_d(transformed, labels)

    # PART E
    visualize_digit(samples[0], 'partD_original')
    for pc_count in [5, 50, 400]:
        matrix_U = np.array([ pair[1] for pair in eig_pairs[:pc_count]]) #we choose first k=2 cols of eigenvector matrix
        transformed = matrix_U.dot(samples[0].T).T
        reconstructed = transformed.dot(matrix_U)
        visualize_digit(reconstructed, 'partD_%d_PC' % pc_count)

def plot_eigenvalues(eigenpairs):
    plt.figure(figsize=(12, 9), dpi=80)

    eigvals = list(map(lambda pair: pair[0], eigenpairs))

    plt.plot(range(400), eigvals, label='Eigenvals')
    plt.interactive(False)

    plt.xlabel('Eigenvectors')
    plt.ylabel('Eigenvalues')
    # plt.legend(loc='upper right', numpoints = 1)
    plt.title("Eigenvalues of MNIST dataset")
    plt.show()
    # plt.savefig('PERC_%s.png' % movie_name)
    # print(counts)


def plot_part_d(matrix, labels):
    colors = "bgrcmyk"
    for k in range(7):
        plt.plot([matrix[i][0] for i in range(len(matrix)) if int(labels[i]) == k],
                 [matrix[i][1] for i in range(len(matrix)) if int(labels[i]) == k], 'o', label=k, color=colors[k])
        plt.plot([matrix[i][0] for i in range(len(matrix)) if int(labels[i]) == 7],
                 [matrix[i][1] for i in range(len(matrix)) if int(labels[i]) == 7], 'o', label=7, color=(0, 0.5, 0.5))
        plt.plot([matrix[i][0] for i in range(len(matrix)) if int(labels[i]) == 8],
                 [matrix[i][1] for i in range(len(matrix)) if int(labels[i]) == 8], 'o', label=8, color=(0.3, 0.3, 0.3))
        plt.plot([matrix[i][0] for i in range(len(matrix)) if int(labels[i]) == 9],
                 [matrix[i][1] for i in range(len(matrix)) if int(labels[i]) == 9], 'o', label=9, color=(0.7, 0.7, 0.7))

    # plt.legend(loc='upper right', numpoints=1)
    plt.title("PCA")
    plt.show()

def visualize_digit(features, label="unnamed"):
    converted_features = list(map(lambda x: int(float(x)*255), features))
    I = np.array(converted_features).reshape((20, 20)).T

    plt.gray()
    plt.imsave("img/%s_%d.png" % (label, random.randint(1,100)), I)
    # plt.imshow(I)



def convert_csv_to_dataset():
    dataset = {'training': {'x': [], 'y': []},
               'cv': {'x': [], 'y': []}}

    features = []
    labels = []

    with open(filepath_first_4000, 'r') as file:
        for line in csv.reader(file):
            features.append(list(map(lambda x: float(x), line[:-1])))
            labels.append(int(line[-1]))

    # separating into 50% / 50% , Training, CV, respectively
    cv_set_offset = int(len(features) * 0.50)

    dataset['training']['x'] += features[:cv_set_offset]
    dataset['training']['y'] += labels[:cv_set_offset]

    dataset['cv']['x'] += features[cv_set_offset:]
    dataset['cv']['y'] += labels[cv_set_offset:]

    return dataset

main()