# same as poly class with ReLU activation and information dropout

import numpy as np
from information_networks.data_generation import poly_dataset_network
from information_networks.fully_connected import fullyConnectedNN
from sklearn.metrics import accuracy_score


def main():

    n = int(1e4)
    informative_p = 10

    X = np.load("informative_X.npy")
    y = np.load("informative_y.npy")

    min_n_train = 8; max_n_train = 200
    min_p = 10; max_p = 100
    n_points = 16; repetition = 20

    architecture = np.array([40, 20])
    sample_dim = np.arange(min_n_train, max_n_train, n_points)
    feature_dim = np.arange(min_p, max_p, n_points)
    test_points = n - max_n_train

    ###########################################################################
    #                        redundant features
    linear_transf = np.random.randn(informative_p, max_p - informative_p)
    linear_combination_X = np.dot(X, linear_transf)
    X = np.hstack((X, linear_combination_X))

    results = np.zeros((sample_dim.size, feature_dim.size, repetition, 3))
    # 3 = accuracy_score, loss_train, information

    for s in sample_dim:
        idx_n = np.where(s == sample_dim)[0][0]
        for p in feature_dim:
            idx_p = np.where(p == feature_dim)[0][0]
            fc = fullyConnectedNN(architecture, X[:s + test_points -1, :p],
                y[:s + test_points -1])
            for r in range(repetition):
                loss_train, loss_test, y_t, y_p, information = fc.fit(s,
                    "information", beta=10.)
                accuracy = accuracy_score(y_true=y_t,y_pred=y_p)
                results[idx_n, idx_p, r, :] = accuracy, loss_train, information

    np.save("relu_results_redundancy.npy", results)

    ###########################################################################
    #                       uninformative features
    X = np.load("informative_X.npy")
    X = np.hstack((X, np.random.randn(n,  max_p - informative_p)))
    results = np.zeros((sample_dim.size, feature_dim.size, repetition, 3))
    # 3 = accuracy_score, loss_train, information

    for s in sample_dim:
        idx_n = np.where(s == sample_dim)[0][0]
        for p in feature_dim:
            idx_p = np.where(p == feature_dim)[0][0]
            fc = fullyConnectedNN(architecture, X[:s + test_points -1, :p],
                y[:s + test_points -1])
            for r in range(repetition):
                loss_train, loss_test, y_t, y_p, information = fc.fit(s,
                    "information", beta=10.)
                accuracy = accuracy_score(y_true=y_t,y_pred=y_p)
                results[idx_n, idx_p, r, :] = accuracy, loss_train, information

    np.save("relu_results_noise.npy", results)


if __name__ == '__main__':
    main()
