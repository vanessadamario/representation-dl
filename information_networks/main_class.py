import numpy as np
from random import sample
from information_networks.fully_connected import fullyConnectedNN
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def sklearn_dataset(n, p, redund, info, classes):
    return make_classification(n_samples=n, n_features=p, n_informative=info,
                               n_redundant=redund, n_classes=classes)


def classification_function(x):
    """ define a classification function, that depends on the input variable x
    f(x) = ax0 + bx1 + cx3 + (x1+x4)**2 + dx5 + e(x6+x7)**3 + fx8
    """
    a = 0.5; b =2.; c = 0.5; d = 0.5; e = 1.; f=0.5
    return -a*x[:,0]+b*x[:,1]-c*x[:,3]-(x[:,1]+x[:,4])**2-d*x[:,5]+e*(x[:,6]+x[:,7])**2-f*x[:,8]


def make_dataset(n, p):
    # informative features : 10
    # classes : 2
    n_informative = 9
    X = np.random.uniform(-1, 1, size=(n, p))
    fx = classification_function(X[:, :n_informative])
    y = (fx - X[:, n_informative] > 0).astype("int")
    return X, y


def main():
    min_n = 8; max_n = 200
    min_p = 10; max_p = 100
    n_points = 16; repetition = 20
    test_points = int(1e4)
    sample_dim = np.arange(min_n, max_n, n_points)
    feature_dim = np.arange(min_p, max_p, n_points)
    architecture = np.array([20, 10, 4])

    results = np.zeros((n_points, n_points, repetition, 2))

    for r in range(repetition):
        for p in feature_dim:
            print("\nP = " + str(p))
            for n in sample_dim:
                print("N = " + str(n))
                # data, label = sklearn_dataset(n+test_points, p,
                #     redund=p-informative_p, info=informative_p, classes=n_classes)
                data, label = make_dataset(n + test_points, p)
                fc = fullyConnectedNN(architecture, data, label)
                loss_train, loss_test, y_t, y_p = fc.fit(n, network_type="no", beta=1.)
                accuracy = accuracy_score(y_t, y_p)
                print("accuracy: " + str(accuracy))
                results[np.where(n == sample_dim)[0][0],
                    np.where(p == feature_dim)[0][0], r, :] = accuracy, loss_train
                # print("accuracy = "+str(accuracy))
    np.save("results.npy", results)


if __name__ == '__main__':
    main()
