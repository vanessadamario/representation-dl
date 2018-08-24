import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import fully_connected, linear


# train a shallow network with random and non random label
# for random label the activation should coincide with the number of points
# check through a boolean matrix how this activation changes across time
#
# see what happens to the activation function - how many ReLU get active
# see if, once the partition is optimal for training it can give better test results if still trained - check which is the function it finds
# ask yourself which can be a good regularizer - in the direction of stopping earlier/ training faster/ memorizing less



class fullyConnectedNN():
    """fullyConnectedNN class: this class creates a fully connected network,
    once that the architecture and the dataset are specified for classification
    task
    """

    def __init__(self, architecture):
        """ fullyConnectedNN constructor
        Parameters :
            architecture, a list, whose size is the number of layers, each component specify the number of neurons contained in
            the specific layer, input and output included
        """
        self.architecture = architecture    # vector of hidden units


    def initialize_weights(self, small_init=False):
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
        init_weights= []

        for k in range(len(self.architecture)-1):
            bound = np.sqrt(6. / (self.architecture[k] + self.architecture[k+1]))
            layer_weights = np.random.uniform(-bound, bound, (self.architecture[k],
                self.architecture[k+1]))
            if small_init:
                layer_weights *= 1e-2
            init_weights.append(layer_weights)
        return init_weights


    def build(self, init_weights, loss_func):
        """
        This method builds the fully connected network, with no regularization
        and activation function as specified
        Parameters:
            init_weights, the initial value of the weights
            activation, if None linear case, if relu - rectifier lin unit
        Returns:
            loss, tf object cross entropy
            tf.nn.softmax(logits), the output of the network (vector of
            probability to belong to a specific class)
        """

        self.X_tf = tf.placeholder(tf.float64, shape=[None,
            self.architecture[0]])
        if loss_func == "cross":
            self.y_tf = tf.placeholder(tf.int32, shape=[None])
        elif loss_func == "square":
            self.y_tf = tf.placeholder(tf.int32, shape=[None, self.architecture[-1]])
        self.tf_weights = []
        network = self.X_tf

        for w in init_weights[:-1]:
            self.tf_weights.append(tf.Variable(w))
            network = tf.maximum(tf.matmul(network, self.tf_weights[-1]), 0)
            print(network.shape)
        self.tf_weights.append(tf.Variable(init_weights[-1]))
        logits = tf.matmul(network, self.tf_weights[-1])

        if loss_func == "cross":
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_tf, logits=logits))

        elif loss_func == "square":
            loss = tf.losses.mean_squared_error(self.y_tf, logits)

        return loss


    def fit(self, X, y, loss_func="square", n_iter=10000, grad_step=5e-2, check_train_loss=False, check_val_loss=False, check_act=False, check_weights=False, init_weights=None):
        """ This method fit the data set with the fully connected architecture
        Parameters:
            X, numpy.array n samples, d features
            y, numpy.array n outputs, n samples
            loss_func, "square" or "cross"
            n_iter, number of max iterations
            grad_step, gradient descent step
            check_train_loss (bool), if True, the 40 values of the training
            loss are returned, equally distributed between 0 and n_iter
            check_val_loss (bool), if True, the matrices (X,y) are splitted in two, and the test loss values are evaluated on a range equally spaced
            check_act (bool), if True, we evaluate the relu activation for each of the layer where the non linearity is applied
        Returns:
            loss_train_value, cross entropy value on the training set
            loss_val_value, cross entropy value on the test set
            self.coefs
        """
        n, _ = X.shape
        if check_val_loss:
            X_train, X_val = np.split(X, 2)
            y_train, y_val = np.split(y, 2)
        else:
            X_train = X
            y_train = y

        if init_weights is None:
            init_weights = self.initialize_weights()
        cost = self.build(init_weights, loss_func=loss_func)

        optimizer = tf.train.GradientDescentOptimizer(grad_step)
        train = optimizer.minimize(cost)
        init_op = tf.global_variables_initializer()

        step_width_btw_control = 1
        if check_train_loss:
            track_train_loss = np.array([])
        if check_val_loss:
            track_val_loss = np.array([])

        # there are len(self.architecture)-1) linear transformation and the last is linear
        if check_act:
            list_activations = [[] for i in range(len(self.architecture)-2)]

        if check_weights:
            list_weights = [[] for i in range(len(self.architecture)-2)]

        with tf.Session() as sess:
            sess.run(init_op)

            for i in range(n_iter):

                sess.run(train, {self.X_tf: X_train, self.y_tf: y_train})

                if (i % step_width_btw_control == 0 and check_train_loss):
                    track_train_loss = np.append(track_train_loss,
                        sess.run(cost, {self.X_tf: X_train,
                            self.y_tf: y_train}))
                if (i % step_width_btw_control == 0 and check_val_loss):
                    track_val_loss = np.append(track_val_loss, sess.run(cost, {self.X_tf: X_val, self.y_tf:y_val}))

                if (i % step_width_btw_control == 0 and check_act):
                    weights_val = sess.run(self.tf_weights, {self.X_tf: X_train, self.y_tf: y_train})
                    tmp_out = X_train
                    for idx, w in enumerate(weights_val[:-1]):
                        tmp_out = np.maximum(tmp_out.dot(w), 0)
                        list_activations[idx].append(tmp_out > 0)

                if (i % step_width_btw_control == 0 and check_weights):
                    weights_val = sess.run(self.tf_weights, {self.X_tf: X_train, self.y_tf: y_train})
                    for idx, w in enumerate(weights_val[:-1]):
                        list_weights[idx].append(w)

            fit_coefs, loss_train_value = sess.run([self.tf_weights, cost], {self.X_tf: X_train, self.y_tf: y_train})

            if check_val_loss:
                loss_val_value = sess.run(cost, {self.X_tf: X_val, self.y_tf: y_val})

        self.coefs = fit_coefs

        self.track_activations = list_activations if check_act else None
        self.track_weights = list_weights if check_weights else None

        self.train_loss = np.append(track_train_loss, loss_train_value) if check_train_loss else loss_train_value

        if check_val_loss:
            self.val_loss = np.append(track_val_loss, loss_val_value) if check_val_loss else loss_val_value

        return self


    def predict(self, X):

        output = X
        for w in self.coefs[:-1]:
            output = np.maximum(np.dot(output, w), 0)
        output = np.dot(output, self.coefs[-1])

        return np.argmax(output, axis=1)


def generate_dataset(n, layers_shape, law="linear", class_task=True):
    """
    Dataset generation for regression problem
    Parameters:
        n, sample size
        d, feature size
        o, output dimension
        layers_shape, list containing for the ith entry the number of nodes from input to output
        law, relu or linear
    Return:
        X, data matrix
        output, y
    """

    d = layers_shape[0]
    o = layers_shape[-1]
    # put thresholds on the values of y
    X = np.abs(np.random.randn(n, d))
    weights = [np.random.randn(dim_in, dim_out) for dim_in, dim_out in zip(layers_shape[:-1], layers_shape[1:])]

    output = X
    for w in weights:
        if law == "relu":
            output = np.maximum(output.dot(w))
        else:
            output = output.dot(w)
    if class_task:
        index_class = np.argmax(output, axis=1)
    return X, index_class, weights


def def_arch(d, hidd, o):
    return [d] + hidd + [o]


def distance_angle(v1, v2):
    return np.arccos(v1.dot(v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))


def evaluate_angle_dist(W1, W2):
    # the matrix is d times h1, for each hyperplane we evaluate the distance
    return [distance_angle(w1, w2) for w1,w2 in zip(W1.T, W2.T)]


def evaluate_activation_change(activations):
    n, _ = activations[0][0].shape
    tmp_count_all_h = []
    for h_idx in range(len(activations)):
        tmp_count_one_h = []
        for act_prev, act_next in zip(activations[h_idx][:-1], activations[h_idx][1:]):
            tmp_count_one_h.append(np.abs(np.sum(act_next.astype("int") - act_prev.astype("int"))).astype("float") / n)
        tmp_count_all_h.append(tmp_count_one_h)
    return tmp_count_one_h if len(tmp_count_all_h) == 1 else tmp_count_all_h


############## main function - data generation - training & plots ##############


def main():

    n = 10000  # samples
    d = 30  # input features
    o = 5  # number of classes
    hidden_teacher = [3]  # architecture generating data
    # hidden_student = [100]  # model use to fit the data
    loss = "square"

    X, y, true_f = generate_dataset(n, def_arch(d, hidden_teacher, o))

    # y = np.random.choice(np.arange(o), size=y.size)

    if loss == "square":
        tmp = np.zeros((n, o))
        tmp[np.arange(n), y] = 1
    y = tmp

    ######################### FIRST SET OF EXPERIMENTS #########################

    directions_array = []
    activations_array = []
    active_change_array = []
    accuracy_array = []
    track_weights_array = []
    training_loss = []
    validation_loss = []

    n_learn = 200

    n_train_val = float(n_learn) / n

    h_layers = [1, 2]
    nodes_per_layers = [30]

    X_learn, X_test, y_learn, y_test = train_test_split(X, y, train_size=n_train_val)
    np.save("y_learn.npy", y_learn)
    np.save("X_learn.npy", X_learn)
    np.save("true_weights.npy", true_f)

    u, s, vh = np.linalg.svd(X_learn.dot(X_learn.T))
    grad_step = 500 / s[0]


    for h in h_layers:

        print(h)
        FullyNN = fullyConnectedNN(architecture=def_arch(d, nodes_per_layers*h, o))
        init_weights = FullyNN.initialize_weights(small_init=True)

        FullyNN.fit(X_learn, y_learn, n_iter=2000, grad_step=grad_step, check_train_loss=True, check_val_loss=True, check_act=True, check_weights=True, init_weights=init_weights)

        weights = FullyNN.coefs
        activations = FullyNN.track_activations

        #distance between initial values and learned ones
        if len(weights)-1 == 1:
            direction_distance = evaluate_angle_dist(init_weights[0], weights[0])
        else:
            direction_distance = []
            for l in range(len(init_weights)):
                direction_distance.append(evaluate_angle_dist(init_weights[l], weights[l]))
            #print("I am deep, change this part")
        # evaluate how the number of active regions change during training

        y_pred = FullyNN.predict(X_test)
        if loss == "square":
            n_test = X_test.shape[0]
            y_tmp = np.zeros((n_test, o))
            y_tmp[np.arange(n_test), y_pred] = 1
            y_pred = y_tmp

        training_loss.append(FullyNN.train_loss)
        validation_loss.append(FullyNN.val_loss)
        directions_array.append(direction_distance)
        activations_array.append(activations)
        active_change_array.append(evaluate_activation_change(activations))
        accuracy_array.append(accuracy_score(y_test, y_pred))
        track_weights_array.append(FullyNN.track_weights)

        pickle.dump(init_weights, open("init_weights_depth_"+str(h)+".pkl", "wb"))

    pickle.dump(training_loss, open("training_curve.pkl", "wb"))
    pickle.dump(validation_loss, open("validation_curve.pkl", "wb"))
    pickle.dump(directions_array, open("directions.pkl", "wb"))
    pickle.dump(activations_array, open("activation.pkl", "wb"))
    pickle.dump(accuracy_array, open("accuracy.pkl", "wb"))
    pickle.dump(active_change_array, open("active_change.pkl", "wb"))
    pickle.dump(track_weights_array, open("track_weights.pkl", "wb"))



if __name__ == '__main__':
    main()
