from random import sample
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def sklearn_dataset(n, p, redund, info, classes):
    return make_classification(n_samples=n, n_features=p, n_informative=info,
                               n_redundant=redund, n_classes=classes)


def main():
    min_n = 8; max_n = 200
    min_p = 20; max_p = 200
    informative_p = 8; n_classes = 2
    n_points = 16; repetition = 20

    samples_dim = np.arange(min_n, max_n, 2, n_points)
    feature_dim = np.arange(min_p, max_p, 2, n_points)
    architecture = np.array([20, 10, 4])

    results = np.zeros((n_points, n_points, repetition))

    for r in range(repetition):
        for p in feature_dim:
            print("\nP = " + str(p))
            for n in samples_dim:
                print("N = " + str(n))
                data, label = sklearn_dataset(n, p, redund=p-informative_p,
                    info=informative_p, classes=n_classes)
                fc = fullyConnectedNN(architecture, data, label)
                loss_train, loss_test, y_t, y_p = fc.fit(n/2, network_type="no", beta=1.)
                accuracy = accuracy_score(y_t, y_p)
                results[np.where(n == samples_dim)[0][0],
                    np.where(p == feature_dim)[0][0], r] = accuracy, loss_train
                # print("accuracy = "+str(accuracy))
    np.save("results.npy", results)


if __name__ == '__main__':
    main()
