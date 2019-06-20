import numpy as np
import scipy.stats as st
from dtree import DecisionTree
import matplotlib.pyplot as plt

# this code only runs under python3


def read_file(filename, sep=" "):
    """
    Read data from a text file
    :param filename:
    :param sep: field separation character
    :return: 2-D list
    """
    with open(filename, "r") as input_file:
        data = [[int(n) for n in line.rstrip("\n").split(sep)] for line in input_file]
    return data


def bootstrap_replicate(data, labels):
    """
    Sample with replacement from data
    :param data:
    :param labels:
    :return: data and corresponding labels for samples (same lengths as originals)
    """
    len_data = len(data)
    idx = [np.random.randint(1, len_data) for _ in range(len_data)]

    return data[idx], labels[idx]


def zero_one_loss(y, y_prime):
    """
    The zero-one loss function.
    :param y:
    :param y_prime:
    :return: 0 if y = y_prime, 1 otherwise
    """
    return np.asarray(y != y_prime, dtype=int)


if __name__ == "__main__":
    train_data = np.array(read_file("data/spect_train.txt", sep=","))
    train_labels = train_data[:, 0]
    train_data = train_data[:, 1:]
    test_data = np.array(read_file("data/spect_test.txt", sep=","))
    test_labels = test_data[:, 0]
    test_data = test_data[:, 1:]

    # create a map of the attributes so we can retain the original column numbers as the tree splits the data
    attributes = list(range(len(train_data[0])))

    # Do an initial run with the full training dataset
    correct = []
    p_max = 1.0
    level_max = 9
    tree = DecisionTree(
        train_data, train_labels, attributes, p_threshold=p_max, max_level=level_max
    )
    y = tree.classify(test_data)
    print(
        "correct = {}".format(
            sum(np.asarray(y == test_labels, dtype=int)) / len(y) * 100
        )
    )

    # Do 10 runs of 25-round bootstrap training varying the depth of the trees from 1 to 10 levels
    n = 25
    num_depths = 10
    bias = np.zeros(num_depths)
    variance = np.zeros(num_depths)
    accuracy = np.zeros(num_depths)
    depths = np.arange(1, num_depths + 1)
    for depth in depths:
        y = np.zeros((n, len(test_data)))
        # We are assuming that N(x) = 0, so there's no noise. This means y_star = y_t
        y_star = t = test_labels
        for i in range(n):
            boot_data, boot_labels = bootstrap_replicate(train_data, train_labels)
            tree = DecisionTree(
                boot_data, boot_labels, attributes, p_threshold=p_max, max_level=depth
            )
            y[i] = tree.classify(test_data)
        # Under zero-one loss the main prediction is the mode (least squares: mean, absolute loss: median)
        y_m = st.mode(y, 0)[0][0]
        # What's the overall test accuracy of our prediction: correct / (correct + incorrect)
        accuracy[depth - 1] = sum(np.asarray(y_m == y_star, int)) / len(y_star)
        # Bias: average zero-one loss between the optimal and main predictions
        bias[depth - 1] = np.mean(zero_one_loss(y_star, y_m))
        # Variance: average {across examples} of [(+1 if main = optimal, -1 otherwise) *
        #           average {across test datasets} zero-one loss between individual predictions and main prediction
        c2 = (
            np.asarray(y_m == y_star, dtype=int) * 2 - 1
        )  # 1 if y_m == y_star, -1 otherwise
        loss_ym_y = np.array([zero_one_loss(y_m, y_i) for y_i in y])
        variance[depth - 1] = np.mean(c2 * np.mean(loss_ym_y, 0))

    # Plot Bias, Variance and overall accuracy
    plt.plot(depths, bias)
    plt.plot(depths, variance)
    plt.plot(depths, accuracy, ls="--")
    plt.legend(["Bias", "Variance", "Accuracy"])
    plt.xlabel("Tree depth")
    plt.ylabel("Loss %")
    plt.grid(True)
    plt.show()
