# dataset generation for binary classification tasks only

import numpy as np
from sklearn.datasets import make_classification


def sklearn_dataset(n, p, redund, info, classes):
    """ call to sklearn dataset generation function
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    """
    return make_classification(n_samples=n, n_features=p, n_informative=info,
                               n_redundant=redund, n_classes=classes)


def classification_function(x):
    """ define a classification function, that depends on the input variable x
    f(x) = ax0 + bx1 + cx3 + (x1+x4)**2 + dx5 + e(x6+x7)**3 + fx8
    Params :
        x, the point in which we want to evaluate the poylnomial function
    Returns:
        f(x)
    """
    a = 0.5; b =2.; c = 0.5; d = 0.5; e = 1.; f=0.5
    return -a*x[:,0]+b*x[:,1]-c*x[:,3]-(x[:,1]+x[:,4])**2-d*x[:,5]+e*(x[:,6]+x[:,7])**2-f*x[:,8]


def make_dataset(n, p):
    """ this function calls the classification_function, passing to it only the
    number of informative features. For each sample the label is assigned by
    thresholding the difference between f(x) and the x9 features. If negative
    y = 1, otherwise 0
    Params:
        n, total number of samples
        p, total number of features (informative and non informative ones)
    Returns:
        X (np.array), input data
        y (np.array), labels
    """
    n_informative = 9
    X = np.random.uniform(-1, 1, size=(n, p))
    fx = classification_function(X[:, :n_informative])
    y = (fx - X[:, n_informative] > 0).astype("int")
    return X, y


def poly_dataset_network(n, informative_p, h=20):

    """
    Function for the generation of a binary classification dataset. Here we
    generate n samples with a shallow architecture with h hidden nodes. The non
    linearity is a polynomial function of degree two, then followed by a
    linearity. This step is followed by the computation of the softmax.
    We compute the argmax for determining the highest probability (and to assign
    the class to each sample)
    Params:
        n (scalar), total number of points
        informative_p (scalar), number of informative features
        h (scalar), number of node in the hidden layer
    Returns:
        X (np.array), n x informative_p dimensions, the input samples
        y (np.array), n x 1 dimension, vector of labels
    """

    n_classes = 2

    X = np.random.randn(n, informative_p)     # input
    y = np.zeros(n)                           # labels

    # since we want the dataset to be balanced for the two classes, we keep
    # generating points until we reach a satisfatory proportion
    # max tolerance for imbalance (0.50 \pm 0.01)
    while np.logical_or(np.sum(y==1).astype("float")/n > 0.51,
        np.sum(y==1).astype("float")/n < 0.49):

        w1 = np.random.randn(informative_p, h)    # first layer
        w2 = np.random.randn(h, n_classes)        # linearity before softmax

        prod = np.dot(X, w1)**2                   # non linearity
        output = np.dot(prod, w2)
        sm_output = np.zeros_like(output)

        for i in range(n):  #softmax computation
            sm_output[i, 0] = np.exp(output[i, 0])/np.sum(np.exp(output[i, :]))
            sm_output[i, 1] = np.exp(output[i, 1])/np.sum(np.exp(output[i, :]))

        y = np.argmax(sm_output, axis=1)

    print("percentage of y=1: " + str(np.sum(y==1).astype("float")/n))

    np.save("informative_X.npy", X)
    np.save("informative_y.npy", y)
    np.save("w1.npy", w1)
    np.save("w2.npy", w2)

    return X, y
