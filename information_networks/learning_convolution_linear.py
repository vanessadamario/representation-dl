import numpy as np
import tensorflow as tf
import warnings
from sklearn.metrics import accuracy_score
from tensorflow.contrib.layers import flatten, linear, fully_connected


class Model(object):
    def __init__(self):
        tf.reset_default_graph()  # delete all the preallocated variables
        self.X = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.int32)
        self.method = None
        self.features = None
        self.classes = None
        self.saver = None
        self.loss = None
        self.output = None
        self.weights_network = []
        self.curve_loss_val = None
        self.curve_loss_train = None

    def compute_output():
        raise NotImplementedError()

    def fit(self, data, label, lr, convergence_crit=True, tol=1e-5,  max_iter=int(1e4)):
        """
        Parameters :

        data : list of numpy.array, each of dimensions (#samples, #features),
        #samples values vary across the list. Three elements must be contained
        in the list (train, validation)
        label : list of numpy.array (1D array), same structure of data
        lr : learning rate of the neural network, fully batch gradient descent
        is the minimization strategy used
        convergence_crit : (default True) if True, the stopping criterion is
        given by the check of loss values computed on validation set, if False,
        the algorithm runs for a max_iters iterations
        toler : (default 1e-5) convergence criterion number, if the difference
        between two losses on the validation set is less that this value, it
        stops the learning procedure (only in convergence_crit is True)
        max_iter : (default 1e4) maximum number of iterations

        Returns :
        self

        Note, new fields
        sess: session of tensorflow, to be used for prediction tasks
        curve_loss_train : loss curve over training set
        curve_loss_val : loss curve evauated on validation set
        """

        if len(data) != 3 or len(label)!=3:   # check if the data are valid
            raise ValueError("error in the dimensions of input")
        X_train, X_val, X_test = data
        y_train, y_val, y_test = label

        self.features = X_train.shape[1]          # number of features
        self.classes = np.unique(y_train).size  # number of classes

        self.output = self.compute_output(self.features, self.classes)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.output))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train = optimizer.minimize(self.loss)
        init_op = tf.global_variables_initializer()

        curve_loss_train = np.array([])
        curve_loss_val = np.array([])
        # curve_acc_train = np.array([])
        # curve_acc_val = np.array([])

        valid_error = 1e10

        self.saver = tf.train.Saver(max_to_keep=1, filename=self.method)
        with tf.Session() as sess:
            sess.run(init_op)
            count_increment_loss_v = 0
            start_increment_loss = -1
            for iteration in range(max_iter):
                # cross entropy minimization
                sess.run(train, {self.X: X_train, self.y: y_train})

                tmp_loss_train = sess.run(self.loss, {self.X: X_train, self.y: y_train})
                tmp_loss_val = sess.run(self.loss, {self.X: X_val, self.y: y_val})

                curve_loss_train = np.append(curve_loss_train, tmp_loss_train)
                curve_loss_val = np.append(curve_loss_val, tmp_loss_val)

                if valid_error - tmp_loss_val < tol and convergence_crit:
                    if count_increment_loss_v and iteration - start_increment_loss > 1:
                        count_increment_loss_v = 0
                    count_increment_loss_v += 1
                    start_increment_loss = iteration
                    if count_increment_loss_v > 5:
                        print("last iteration: " + str(iteration))
                        break
                valid_error = tmp_loss_val
                if iteration == max_iter - 1 and convergence_crit:
                    warnings.warn("Not converged")
            self.weights_network = [sess.run(v)
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
            self.saver.save(sess, self.method)
            sess.close()

        self.curve_loss_val = curve_loss_val
        self.curve_loss_train = curve_loss_train

        return self

    def predict(self, X_test):
        """
        Prediction using the model obtained using fit function
        Parameters
        X_test: numpy.array to use as a test set
        Returns
        y_pred: numpy.array with predicted classes for the test set
        """
        with tf.Session() as sess:
            self.saver.restore(sess, self.method)
            y_pred = sess.run(tf.argmax(tf.nn.softmax(tf.reshape(self.output,
                shape=[X_test.shape[0], self.classes])), 1), {self.X: X_test})
            sess.close()
        return y_pred

    def loss_value(self, X, y):
        """
        Evaluation of the loss function over a generic set (for the test set in
        particular)
        Parameters
        X: numpy.array
        y: numpy.array
        Returns
        loss_value
        """
        with tf.Session() as sess:
            self.saver.restore(sess, self.method)
            loss_value = sess.run(self.loss, {self.X: X, self.y: y})
            sess.close()
        return loss_value


class Linear(Model):
    """docstring for Linear."""
    def __init__(self):
        super(Linear, self).__init__()
        self.method="linear"

    def compute_output(self, features, classes):
        return linear(tf.reshape(self.X, [-1, 1, features]), classes)

    def fit(self, data, label, lr, convergence_crit, tol, max_iter):
        super(Linear, self).fit(data, label, lr, convergence_crit, tol, max_iter)
    # in the linear case the attribute weights_network is a list
    # whose first element contains the weights, the second the bias terms

class Convolutional(Model):
    """docstring forConvolutional."""
    def __init__(self, n_filter, length_filter, stride, init_conv_weights=None, init_linear_weights=None):
        super(Convolutional, self).__init__()
        self.n_filter = n_filter
        self.length_filter = length_filter
        self.stride = stride
        self.init_conv_weights = init_conv_weights
        self.init_linear_weights = init_linear_weights
        self.method = "convolutional"
        self.conv_weights = None

    def compute_output(self, features, classes):
        if self.init_conv_weights is not None:
            if self.init_conv_weights.shape[0] != self.length_filter or self.init_conv_weights.shape[1]!=self.n_filter:
                raise ValueError("init matrix dimensions do not match")
            self.init_conv_weights = self.init_conv_weights.reshape(
                self.length_filter, 1, self.n_filter).astype("float32")
        else:
            x = 1./sqrt(6 * self.n_filter * self.length_filter)
            self.init_conv_weights = np.random.uniform(-x, x, size=(self.length_filter, 1, self.n_filter)).astype("float32")

        if self.init_linear_weights is not None:
            if self.init_linear_weights.shape[0] != (self.n_filter * self.features) or self.init_linear_weights.shape[1]!=self.classes:
                raise ValueError("init matrix dimensions do not match")
            self.init_linear_weights = self.init_linear_weights.reshape(
                self.features * self.n_filter, 1, self.classes).astype("float32")

        self.conv_weights = tf.Variable(self.init_conv_weights)
        convolution = tf.nn.conv1d(self.X, filters=self.conv_weights,
            stride=self.stride, padding="SAME")
        flat_convolution = flatten(tf.transpose(convolution, perm=[0, 2, 1]))
        output = fully_connected(tf.reshape(flat_convolution,
            [-1, 1, self.n_filter * self.features]), num_outputs=self.classes,
            activation_fn=None, weights_initializer=tf.constant_initializer(self.init_linear_weights))
        # output = fully_connected(tf.reshape(flat_convolution, [-1, 1, self.n_filter * features]),
        #     num_outputs=self.classes, activation_fn=None,
        #     weights_initializer=tf.constant_initializer(init_linear_weights))
        return output

    def fit(self, data, label, lr, convergence_crit, tol,  max_iter):
        super(Convolutional, self).fit(data, label, lr, convergence_crit, tol, max_iter)
        return self
