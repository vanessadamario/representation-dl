import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import fully_connected, linear, dropout


class fullyConnectedNN():
    """fullyConnectedNN class: this class creates a fully connected network,
    once that the architecture and the dataset are specified for classification
    task. The methods of the class build different networks (non regularized
    ones, dropout and information dropout regularizers).
    """

    def __init__(self, architecture, X, y):
        """ fullyConnectedNN constructor
        Parameters :
            architecture, a numpy array, whose size is the number of hidden
            layers, each component specify the number of neurons contained in
            the specific hidden layer
            X, np.array of dimension n x p, n #samples, p #feature space
            y, np.array of labels
        """
        self.architecture = architecture    # vector of hidden units
        self.X = X                          # input data
        self.y = y


    def build_placeholders(self):
        """ Here we build placeholders. The methods creates new tensorflow
        objects of the fullyConnectedNN class, which are necessary for the
        fit method
        # Parameters :
        #     training_size, number of training points
        """
        # n_classes = np.unique(self.y).size
        self.X_tensorflow = tf.placeholder(tf.float64, shape=[None,
            self.X.shape[1]])
        self.y_tensorflow = tf.placeholder(tf.int32, shape=[None])


    def initialize_weights(self):
        """ This method creates a set of weights, given the architecture, the
        dimension of the feature space and the number of classes. This weights
        are sampled from a uniformly random distribution, following the work
        `Efficient backprop` - LeCun.
        For every layer of the network we append the set of initial weights to
        a list
        Returns:
            init_weights, the list where each element is a np.array containing
            the initial value of the weights for a specific layer
        """
        n_classes = np.unique(self.y).size
        init_weights= []

        layers = np.append(np.append(self.X.shape[1], self.architecture),
            n_classes)
        for k in range(layers.size-1):
            bound = np.sqrt(6. / (layers[k] + layers[k+1]))
            layer_weights = np.random.uniform(-bound, bound, (layers[k],
                layers[k+1]))
            init_weights.append(layer_weights)
        return init_weights


    def build_no_regularization(self, init_weights):
        """
        This method builds the fully connected network, with no regularization
        Parameters:
            init_weights, the initial value of the weights
        Returns:
            loss, tf object cross entropy
            tf.nn.softmax(logits), the output of the network (vector of
            probability to belong to a specific class)
        """

        n_classes = np.unique(self.y).size  # this gives the number of classes
        network = self.X_tensorflow
        # at the first iteration the network corresponds to the input data
        # we iteratively map the network to the output through a linear
        # linear transformation followed by a non linearity (ReLU by default)
        for l in range(self.architecture.size):
            network = fully_connected(network, num_outputs=self.architecture[l],
                weights_initializer=tf.constant_initializer(init_weights[l]))
        # then we compute logits, so we use a final linear transformation
        logits = linear(network, n_classes,
            weights_initializer=tf.constant_initializer(init_weights[-1]))
        # we compute the loss function, by doing the mean of the cross entropy
        # given X, y
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_tensorflow, logits=logits))
        return loss, tf.nn.softmax(logits)


    def build_bernoulli_dropout(self, init_weights, p_bernoulli, where=-1):
        """
        This method creates a fully connected network with regularization
        (dropout) `Dropout: A Simple Way to Prevent Neural Networks from
        Overfitting` - Srivastava.
        Parameters:
            init_weights, initial weights values
            p_bernoulli, probability of killing a neuron. It follows a
            bernoullian distribution
            where, the hidden layer to which apply the dropout, by default it is
            the last one
        Returns:
            loss, tf object cross entropy
            tf.nn.softmax(logits), the output of the network (vector of
            probability to belong to a specific class)
        """
        n_classes = np.unique(self.y).size
        # if where == -1 we want to regularize the last layer. For this scope we
        # need to know which is the index relative to the last transformation
        if where == -1:
            where = self.architecture.size - 1
        network = self.X_tensorflow
        for i in range(self.architecture.size):
            network = fully_connected(network, num_outputs=self.architecture[i],
                weights_initializer=tf.constant_initializer(init_weights[i]))
            # if the layer is the one we want to regularize we add the dropout
            if i == where:
                network = dropout(network, keep_prob=p_bernoulli)
        logits = linear(network, n_classes,
            weights_initializer=tf.constant_initializer(init_weights[-1]))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_tensorflow, logits=logits))
        return loss, tf.nn.softmax(logits)


    def sample_log_normal(self, mean, sigma):
        """ This function creates points log-normally distributed :
        exp (mean + sigma * N(0, 1)), with N(0, 1) normal distribution
        """
        e = tf.random_normal(tf.shape(mean), mean=0., stddev=1., dtype=tf.float64)
        return tf.exp(mean + sigma * e)


    def build_information_dropout(self, init_weights, beta=1., max_alpha=0.7, where=-1):
        """ This function builds a network with information dropout regularizer
        `Information Dropout: Learning Optimal Representations Through Noisy
        Computation` - Achille, Soatto

        Parameters:
            init_weights, np.array of initial weights values
            beta, regularization parameter constant
            max_alpha, maximum value for the sample_log_normal distribution
            where, layer to which apply the regularization dropout, by default
            it is the last one
        Returns:
            loss, the loss function value (cross entropy)
            kl, information value
            tf.nn.softmax(logits), vector of probability related to a specific
            class
        """
        n_classes = np.unique(self.y).size
        if where == -1:
            where = self.architecture.size - 1
        # again we assign to network the input data, these will be then
        # transformed mapped in the next iterations
        network = self.X_tensorflow
        for i in range(self.architecture.size):
            network = fully_connected(network, num_outputs=self.architecture[i],
                weights_initializer=tf.constant_initializer(init_weights[i]))
            # in particular, for the layer we want to regularize
            if i == self.architecture.size - where:
                # we need to multiply the original network for distribution
                # we choose here the sigmoid activation, because we want this
                # output to have plausible values (it will be interpreted as a
                # variance next)
                alpha = max_alpha * fully_connected(network,
                    num_outputs=self.architecture[i], activation_fn=tf.nn.sigmoid)
                # kl is the kullbach leibler divergence
                kl = - 0.5 * tf.reduce_mean(tf.log(alpha / (max_alpha + 1e-3)))
                eps = self.sample_log_normal(tf.zeros_like(alpha), alpha)
                # here we blur our network with noise
                network =  network * eps

        logits = linear(network, n_classes,
            weights_initializer=tf.constant_initializer(init_weights[-1]))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_tensorflow, logits=logits))
        return loss, kl, tf.nn.softmax(logits)


    def fit(self, n_train, network_type="no", beta=0., p_drop=0.5, n_iters=int(2e3), grad_step=1e-2, where=-1):
        """ This method fit the data set with the fully connected architecture
        Parameters:
            n_train, # training samples
            network_type, "no" - fully connected network without regularization,
            "dropout" fully connected network with bernoullian dropout,
            "information" fully connected network with information dropout
            beta, regularization parameter (iff network_type=="information")
            p_drop, bernoullian probability of dropout (iff network_type=="dropout")
            n_iters, number of max iterations
            grad_step, gradient descent step
            where, where to apply the regularizer, by default to the last layer
        Returns:
            loss_train_value, cross entropy value on the training set
            loss_test_value, cross entropy value on the test set
            y_test, true y values for the test set
            y_pred, predicted y valued for the X test set
            information, information value, computed through KL divergence
        """
        n, d = self.X.shape
        n_classes = np.unique(self.y).size
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
            train_size=float(n_train) / n) # split train and test sets

        # we call init_weights and then we pass this matrix to all the following
        # methods in order to get a fair comparison of the different network_types
        init_weights = self.initialize_weights()
        self.build_placeholders()

        if network_type == "no":
            loss, pred = self.build_no_regularization(init_weights)
            beta, information = np.zeros(2)

        elif network_type == "information":
            loss, information, pred = self.build_information_dropout(init_weights, beta,
                where)

        elif network_type == "dropout":
            loss, pred = self.build_bernoulli_dropout(init_weights, p_drop,
                where)
            beta, information = np.zeros(2)

        else:
            print("input error, this option is not implemented")
            exit(-1)

        cost = loss + beta * information  # cost function to be minimized
        optimizer = tf.train.GradientDescentOptimizer(grad_step)  # fully batch GD
        train = optimizer.minimize(cost)  # minimization of the cost function
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(n_iters):
                sess.run(train, {self.X_tensorflow: X_train,
                    self.y_tensorflow: y_train})
            loss_train_value = sess.run(loss, {self.X_tensorflow: X_train,
                self.y_tensorflow: y_train})

            if network_type == "information":
                information_value = sess.run(information, {self.X_tensorflow:
                    X_train, self.y_tensorflow: y_train})

            loss_test_value = sess.run(loss, {self.X_tensorflow: X_test,
                self.y_tensorflow: y_test})
            y_pred = sess.run(tf.argmax(pred, 1), {self.X_tensorflow: X_test})

        if network_type == "information":
            return loss_train_value, loss_test_value, y_test, y_pred, information

        else:
            return loss_train_value, loss_test_value, y_test, y_pred
