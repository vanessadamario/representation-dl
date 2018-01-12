import numpy as np
from information_networks.signal_generation import signal_gen
from information_networks.learning_convolution_linear import Convolutional, Linear
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def main():

    # path_images = "./images/"
    sample_size = int(1e4); features = 1000; n_classes = 3
    p_pattern = np.array([3, 4, 2]); length_patterns = 50
    noise_ar = np.array([0])  # np.linspace(0., 3., 10)

    repetition = 1  # 20
    results_values = 3  # train loss, test loss, accuracy score
    results_linear = np.zeros((noise_ar.size, repetition, results_values))
    results_convolution = np.zeros((noise_ar.size, repetition, results_values))

    for noise_val in noise_ar:
        idx_noise = np.where(noise_val == noise_ar)[0][0]
        for r in range(repetition):
            noiseless_X, y, patterns = signal_gen(sample_size, features, n_classes,
                p_pattern, length_patterns, noise=0.)

            X =  noiseless_X + noise_val * np.random.randn(noiseless_X.shape[0],
                features)
            X = np.reshape(X, newshape=(X.shape[0], X.shape[1], 1))

            #   training - test sets
            train_val_points = 400
            perc_train = float(train_val_points)/sample_size

            sss_learn_test = StratifiedShuffleSplit(n_splits=2,
                train_size=perc_train, test_size=1-perc_train)

            idx_split_lt, _ = sss_learn_test.split(X, y)
            idx_learn = idx_split_lt[0]; idx_test = idx_split_lt[1]

            X_learn = X[idx_learn, :]; y_learn = y[idx_learn]
            X_test = X[idx_test, :]; y_test = y[idx_test]

            sss_train_valid = StratifiedShuffleSplit(n_splits=2,
                train_size=0.5, test_size=0.5)

            idx_split_tv, _= sss_train_valid.split(X_learn, y_learn)
            idx_train = idx_split_tv[0]; idx_valid = idx_split_tv[1]

            X_train = X_learn[idx_train, :]; y_train = y_learn[idx_train]
            X_valid = X_learn[idx_valid, :]; y_valid = y_learn[idx_valid]

            dataset = [X_train, X_valid, X_test]
            labels = [y_train, y_valid, y_test]
            #   convolutional model characteristics

            n_filters = 10
            length_filter = 50
            n_iters = int(2e3)

            linearModel = Linear()
            linearModel.fit(dataset, labels, lr=1e-2, convergence_crit=1e-4, tol=1e-7,  max_iter=int(1e4))
            loss_training = linearModel.curve_loss_train
            loss_validation = linearModel.curve_loss_val

            predicted_y = linearModel.predict(X_test)
            accuracy = accuracy_score(y_test, predicted_y)
            # results_linear[idx_noise, r, :] = accuracy, loss_train, loss_test
            print("\nLINEAR MODEL CONVERGENCE CONTROL")
            # # print("loss values on training set: "+str(np.mean(loss_train)))
            # print("loss values on test set: "+str(np.mean(loss_test)))
            print("accuracy over test set: "+str(accuracy)+"; chance 0.33")
            plt.plot(loss_training, label="training set")
            plt.plot(loss_validation, label="validation set")
            plt.ylabel("loss values")
            plt.xlabel("iterations")
            plt.legend()
            plt.show()
            plt.close()
            linear_weights = linearModel.weights_network
            print("len linear weights", len(linear_weights))
            for i in range(len(linear_weights)):
                print(linear_weights[i].shape)

            print("\nCONVOLUTIONAL MODEL")
            convModel = Convolutional(n_filter=10, length_filter=50, stride=1)
            convModel.fit(dataset, labels, lr=1e-2, convergence_crit=1e-4, tol=1e-7,  max_iter=int(1e4))
            loss_training = convModel.curve_loss_train
            loss_validation = convModel.curve_loss_val
            conv_weights = convModel.weights_network

            predicted_y = convModel.predict(X_test)
            accuracy = accuracy_score(y_test, predicted_y)
            print("accuracy over test set: "+str(accuracy)+"; chance 0.33")

            # np.save("filters.npy", filters)
            # np.save("patterns.npy", np.array(patterns))
            #
            # # results_convolution[idx_noise, r, :] = accuracy, loss_train, loss_test

            # print("loss values on training set: "+str(np.mean(loss_train)))
            # print("loss values on test set: "+str(np.mean(loss_test)))
            # print("accuracy over test set: "+str(accuracy)+"; chance 0.33")



            ############################################################################
            #                       plots experiments with repetition
            # if not r:
            #     for i in range(len(patterns)):
            #         for j in range(p_pattern[i]):
            #             plt.plot(patterns[i][j, :])
            #             plt.title("pattern from class " + str(i+1))
            #             plt.savefig(path_images + "class_" + str(i+1) + "_pattern_" + str(j+1) + ".png")
            #             plt.close()
            #
            #     for i in range(n_classes):
            #         plt.plot(X[i * sample_size / n_classes, :])
            #         plt.plot(X[i * sample_size / n_classes + 1, :])
            #         plt.title("examples from class "+str(i+1))
            #         plt.savefig(path_images + "noise_" + str(noise_val) + "_signals_class_" + str(i+1) + ".png")
            #         plt.close()
            #
            #     for i in range(n_filters):
            #         plt.plot(filters[:, 0, i])
            #         plt.title("learned filter " + str(i+1))
            #         plt.savefig(path_images + "noise_" + str(noise_val) + "_filter_" + str(i+1) + ".png")
            #         plt.close()


        # warning : this is useful if the n_classes == 3
        # warning : this works only for the specifics given for the dataset & model
        # f2, axarr = plt.subplots(2, 10)
        # for i in range(n_filters):
        #     if i < p_pattern[0]:
        #         axarr[0, i].plot(patterns[0][i, :], 'r')
        #     elif np.logical_and(i >= 3, i < 7):
        #         axarr[0, i].plot(patterns[1][i-3, :], 'b')
        #     elif np.logical_and(i >= 7, i < 9):
        #         axarr[0, i].plot(patterns[2][i-7, :], 'g')
        #
        #     axarr[1, i].plot(filters_values[:, 0, i], label="filters "+str(i))
        # plt.legend()
        # plt.savefig("truePatterns_convWeights.png")
        # plt.show()
        # plt.close()

    # np.save("results_linear.npy", results_linear)
    # np.save("results_convolution.npy", results_convolution)

    return


if __name__ == '__main__':
    main()
