import numpy as np
from sys import argv
from information_networks.data_generation import poly_dataset_network
from information_networks.fully_connected import fullyConnectedNN
from sklearn.metrics import accuracy_score


def main(argv):

    X = np.load("redundant_X.npy")
    y = np.load("informative_y.npy")
    n = X.shape[0]
    informative_p = 10

    architecture = 80

    n_samples = int(argv[1])
    n_features = int(argv[2])
    max_n_train = 200

    repetition = 2

    values = 2  # accuracy, loss on training set
    results = np.zeros((repetition, values))
    track_loss_train_repetition = []

    test_points = n - max_n_train
    fc = fullyConnectedNN(architecture, X[:n_samples + test_points -1, :n_features],
        y[:n_samples + test_points -1])

    for r in range(repetition):
        loss_train, loss_test, y_t, y_p, track_loss_train = fc.fit(n_samples,
            "poly", check_training_loss=True)
        accuracy = accuracy_score(y_true=y_t,y_pred=y_p)
        results[r, :] = accuracy, loss_train
        track_loss_train_repetition.append(track_loss_train)
    track_loss_train_repetition = np.array(track_loss_train_repetition)
    np.save("track_loss_"+argv[1]+"_"+argv[2]+".npy", track_loss_train_repetition)
    np.save("results_"+argv[1]+"_"+argv[2]+".npy", results)


if __name__ == '__main__':
    main(argv)
