import numpy as np
from random import sample
from information_networks.data_generation import make_dataset
from information_networks.fully_connected import fullyConnectedNN
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def main():
    min_n = 8; max_n = 200
    min_p = 10; max_p = 100
    n_points = 16; repetition = 20
    test_points = int(1e4)
    sample_dim = np.arange(min_n, max_n, n_points)
    feature_dim = np.arange(min_p, max_p, n_points)
    architecture = np.array([20, 10, 4])

    results = np.zeros((sample_dim.size, feature_dim.size, repetition, 2))

    for r in range(repetition):
        for p in feature_dim:
            print("\nP = " + str(p))
            for n in sample_dim:
                print("N = " + str(n))
                data, label = make_dataset(n + test_points, p)
                fc = fullyConnectedNN(architecture, data, label)
                loss_train, loss_test, y_t, y_p = fc.fit(n, network_type="no", beta=0.)
                accuracy = accuracy_score(y_t, y_p)
                print("accuracy: " + str(accuracy))
                results[np.where(n == sample_dim)[0][0],
                    np.where(p == feature_dim)[0][0], r, :] = accuracy, loss_train
                # print("accuracy = "+str(accuracy))
    np.save("results.npy", results)


if __name__ == '__main__':
    main()
