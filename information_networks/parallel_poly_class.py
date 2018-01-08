import numpy as np
from sys import argv
from information_networks.data_generation import poly_dataset_network
from information_networks.fully_connected import fullyConnectedNN
from sklearn.metrics import accuracy_score


def main(argv):

    X = np.load("redundant_X.npy")  # redundant case
    # X = np.load("noisy_X.npy")  # noisy case
    y = np.load("informative_y.npy")
    n = X.shape[0]
    informative_p = 10

    architecture = 40

    n_samples = int(argv[1])
    n_features = int(argv[2])
    max_n_train = 200

    repetition = 20
    save_path = "./redundantX_architecture40_moreIters/"  # path for redundant features
    # save_path = "./noisyX_architecture40/"  # path for noisy features

    check_train = True
    check_test = False
    values = 3  # accuracy, loss on training set, loss on test set
    max_iters = int(4e3)
    results = np.zeros((repetition, values))
    track_train_repetition = []

    test_points = n - max_n_train
    fc = fullyConnectedNN(architecture, X[:n_samples + test_points -1, :n_features],
        y[:n_samples + test_points -1])

    for r in range(repetition):
        loss_train, loss_test, y_t, y_p, track_train = fc.fit(n_samples,
            "poly", n_iters=max_iters, check_training_loss=check_train,
            check_test_loss=check_test)
        accuracy = accuracy_score(y_true=y_t,y_pred=y_p)
        results[r, :] = accuracy, loss_train, loss_test
        track_train_repetition.append(track_train)
    track_train_repetition = np.array(track_train_repetition)

    np.save(save_path + "track_train" + argv[1] + "_" + argv[2] + ".npy",
        track_train_repetition)
    np.save(save_path + "results_" + argv[1] + "_" + argv[2] + ".npy",
        results)


if __name__ == '__main__':
    main(argv)
