import numpy as np
from information_networks.data_generation import poly_dataset_network
from information_networks.fully_connected import fullyConnectedNN
from sklearn.metrics import accuracy_score


def main():

    n = int(1e4)
    informative_p = 10

    X, y = poly_dataset_network(n,informative_p, h=20)

    min_n_train = 8; max_n_train = 200
    min_p = 10; max_p = 100
    n_points = 16; repetition = 20

    # we add the non informative features here
    X = np.hstack((X, np.random.randn(n,  max_p - informative_p)))
    print(X.shape)

    architecture = 80
    sample_dim = np.arange(min_n_train, max_n_train, n_points)
    # training set dimension
    feature_dim = np.arange(min_p, max_p, n_points)  # feature set dimension
    test_points = n - max_n_train

    results = np.zeros((sample_dim.size, feature_dim.size, repetition, 2))
    # 2 = accuracy_score, loss_train

    for s in sample_dim:
        idx_n = np.where(s == sample_dim)[0][0]
        for p in feature_dim:
            idx_p = np.where(p == feature_dim)[0][0]
            fc = fullyConnectedNN(architecture, X[:s + test_points -1, :p],
                y[:s + test_points -1])
            for r in range(repetition):
                loss_train, loss_test, y_t, y_p = fc.fit(s, "poly")
                accuracy = accuracy_score(y_true=y_t,y_pred=y_p)
                results[idx_n, idx_p, r, :] = accuracy, loss_train

    np.save("poly_results.npy", results)


if __name__ == '__main__':
    main()
