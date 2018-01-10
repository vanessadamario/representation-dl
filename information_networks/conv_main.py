# TODO: track the values of the loss during traning (early stopping ?)
#     : the initialization of the weights seems not to affect dramatically the
#       results, we always check the convergence of the training loss

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.contrib.layers import flatten, linear
from information_networks.signal_generation import signal_gen
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def main():

    ############################################################################
    #                           specifics
    #   dataset
    sample_size = int(1e4); features = 1000; n_classes = 3
    p_pattern = np.array([3, 4, 2]); length_patterns = 50
    X, y, patterns = signal_gen(sample_size, features, n_classes, p_pattern,
        length_patterns, noise=0.1)

    X = np.reshape(X, newshape=(X.shape[0], X.shape[1], 1))
    # plt.plot(X[0, :])
    # plt.plot(X[1, :])
    # plt.plot(X[2, :])
    # plt.show()
    #   training - test sets
    train_points = 200
    perc_train = float(train_points)/sample_size
    sss = StratifiedShuffleSplit(n_splits=2, train_size=perc_train,
        test_size=1-perc_train)
    idx_split, _ = sss.split(X, y)
    idx_train = idx_split[0]; idx_test = idx_split[1]
    X_train = X[idx_train, :]; y_train = y[idx_train]
    X_test = X[idx_test, :]; y_test = y[idx_test]

    #   convolutional model characteristics
    n_filters = 10
    length_filter = 50
    n_iters = int(1e3)

    #   build placeholders
    signals = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.int32)

    ############################################################################
    #                      convolutional algo
    # this is equivalent to make an average on the length_filter window
    init_conv_weights = 1./ np.sqrt(length_filter) * np.ones((
        length_filter, 1, n_filters)).astype("float32")
    conv_weights = tf.Variable(init_conv_weights)

    convolution = tf.nn.conv1d(signals, filters=conv_weights, stride=1,
        padding="SAME")
    flat_convolution = flatten(tf.transpose(convolution, perm=[0, 2, 1]))
    logits = linear(tf.reshape(flat_convolution, [-1, 1, n_filters * features]),
        n_classes)
    # add a linear transformation which takes as input flat_convolution and
    # with output of dimension (batch, n_classes)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
        logits=logits)
    # loss function with softmax computed internally
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    # optimize it with gradient descent or other methods (Adam, SGD)
    train = optimizer.minimize(loss)
    # train the model

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for n in range(n_iters):
            sess.run(train, feed_dict={signals: X_train, labels: y_train})
        filters_values = sess.run(conv_weights)
        train_loss_value = sess.run(loss, feed_dict={signals: X_train,
            labels: y_train})
        test_loss_value = sess.run(loss, feed_dict={signals: X_test,
            labels: y_test})
        y_pred = sess.run(tf.argmax(tf.nn.softmax(tf.reshape(logits,
            shape=[y_test.shape[0], n_classes])), 1), {signals:X_test})
        print("convolution shape", sess.run(tf.shape(convolution), {signals:X_train}))
    accuracy = accuracy_score(y_test, y_pred)
    print("\nCONVOLUTIONAL MODEL")
    print("loss values on training set: "+str(np.mean(train_loss_value)))
    print("loss values on test set: "+str(np.mean(test_loss_value)))
    print("accuracy over test set: "+str(accuracy)+"; chance 0.33")
    ############################################################################
    #                       linear algo
    init_linear_weights = tf.random_normal([features, n_classes])
    linear_output = linear(tf.reshape(signals, [-1, 1, features]), n_classes)
    loss_linear_model = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=linear_output)
    optimizer_linear_model = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
    train_linear_model = optimizer_linear_model.minimize(loss_linear_model)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for n in range(n_iters):
            sess.run(train_linear_model, feed_dict={signals: X_train,
                labels: y_train})
        train_loss_value = sess.run(loss_linear_model,
            feed_dict={signals: X_train, labels: y_train})
        test_loss_value = sess.run(loss, feed_dict={signals: X_test,
            labels: y_test})
        y_pred = sess.run(tf.argmax(tf.nn.softmax(tf.reshape(linear_output,
            shape=[y_test.shape[0], n_classes])), 1), {signals:X_test})
    accuracy = accuracy_score(y_test, y_pred)
    print("\nLINEAR MODEL")
    print("loss values on training set: "+str(np.mean(train_loss_value)))
    print("loss values on test set: "+str(np.mean(test_loss_value)))
    print("accuracy over test set: "+str(accuracy)+"; chance 0.33")

    ############################################################################
    #                       plots
    # warning : this is useful if the n_classes == 3
    f1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.plot(X[0, :], label="class 1")
    ax1.plot(X[1, :])
    ax2.plot(X[sample_size/2, :], label="class 2")
    ax2.plot(X[sample_size/2+1, :])
    ax3.plot(X[-2, :], label="class 3")
    ax3.plot(X[-1, :])
    plt.savefig("signals.png")
    plt.close()

    # warning : this works only for the specifics given for the dataset & model
    f2, axarr = plt.subplots(2, 10)
    for i in range(n_filters):
        if i < p_pattern[0]:
            axarr[0, i].plot(patterns[0][i, :], 'r')
        elif np.logical_and(i >= 3, i < 7):
            axarr[0, i].plot(patterns[1][i-3, :], 'b')
        elif np.logical_and(i >= 7, i < 9):
            axarr[0, i].plot(patterns[2][i-7, :], 'g')

        axarr[1, i].plot(filters_values[:, 0, i], label="filters "+str(i))
    plt.legend()
    plt.savefig("truePatterns_convWeights.png")
    plt.show()
    plt.close()

    return


if __name__ == '__main__':
    main()
