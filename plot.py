import matplotlib.pyplot as plt

def plot_barchart():
    # c_vals = ["0.0001", "0.001", "0.01", "0.1", "1", "10", "100", "1000"]
    c_vals = ["0.0001", "0.0003", "0.0005", "0.0008", "0.001", "0.01", "0.1", "1", "10", "100", "1000"]
    # vals = [91.48, 93.37, 96.2, 95.89, 95.58, 95.89, 95.89]
    vals = [50.47, 73.7, 87.45, 90.12, 90.8, 94.6, 95.7, 94.9, 94.9, 94.9, 94.9,]
    plt.barh(range(11), vals, align='center')
    plt.yticks(range(11), c_vals)
    plt.xticks(range(50, 100, 2))
    plt.axis([50, 100, -1, 12, ])

    plt.xlabel('Accuracy rate')
    plt.ylabel('C values')
    # plt.legend(loc='upper right', numpoints = 1)
    plt.title("Accuracy of Linear SVM over different C parameters")
    plt.show()

plot_barchart()