# TODO: gaussian dilation

import numpy as np


def signal_gen(n, p, n_classes, p_pattern, length_patterns, noise):
    """
    function for the generation of one dimensional signals for classification
    tasks. We will generate samples belonging to different classes. each class
    is characterized by a number of patterns (fixed by the user)
    These patterns will recur through the signals
    Parameters:
        n (int), number of signals
        p (int), length of the signals
        n_classes (int), #classes
        p_pattern (np.array), each element corresponds to the #characteristic
            for that class
        length_patterns (int), length for each pattern, equivalent to all
        noise (float), percentage (with respect to the squared amplitude of the
            signal), of gaussian additive noise
    Returns:
        X (np.array), dim n x p, samples
        y (np.array), dim n x 1, labels
        patterns (list of (np.array)),
            dim (n_classes*[p_pattern, length_patterns]) characteristic patterns
        w (list of (np.array)),
            dim (n_classes*[p_pattern, p - length_patterns])
    """

    if n_classes != p_pattern.size:
        print("error, #classes and elements specified in the array don't match")
        exit(-1)

    X = np.zeros((n, p))  # initialized as zero
    y = np.zeros(n)

    # here we create the patterns. each class has its own frequencies.
    # each pattern is generated using a sinusoidal function of these f
    # the pattern is then amplified randomly (multiplied with a constant)
    patterns = []


    lowest_f = np.log10(1. / length_patterns)   # in log scale
    highest_f = np.log10(1. / 4)
    # half of the period for the lowest, 1./4 four points for the latter
    possible_f = np.logspace(lowest_f, highest_f, n_classes*3)
    possible_f = np.split(possible_f, n_classes)
    mu = 1.  # mean value of the amplitude for each pattern
    sigma = 0.1
    for k in range(n_classes):
        f = np.random.choice(possible_f[k], size=p_pattern[k])
        # generate a sinusoidal signal, different for the different classes
        sinusoidal = np.zeros((p_pattern[k], length_patterns))
        for j in range(p_pattern[k]):
            sinusoidal[j, :] = np.sin(2 * np.pi * f[j] * np.arange(length_patterns))
        tmp_const = 1 #  mu + np.random.randn(p_pattern[k])
        patterns.append(np.multiply(tmp_const, sinusoidal.T).T)

    # here we create the signal. the patterns will be shifted through the signal
    # we then need to split p in a number of segment, then we will randomly
    # select the segment where we want the pattern to appear
    segments = p / length_patterns
    samples_per_class = n / n_classes  # sample points for each class

    all_x_in_signal = np.arange(0, segments-1, 2)  # this vector contains all
    # the segments (with step=2), to avoid overlap
    all_x_in_segment = np.arange(0, length_patterns)  # all positions in the segment
    # for each pattern we put a n_patterns_per_signal
    n_patterns_per_signal = segments/2
    # we will extract the position using the
    idx_all_pos = np.arange(n_patterns_per_signal)

    for k in range(n_classes):
        matrix_x_in_segment = np.random.choice(all_x_in_segment,
            size=(samples_per_class, n_patterns_per_signal))

        matrix_x_in_signal = np.array([]).reshape(0, n_patterns_per_signal)
        for j in range(samples_per_class):
            tmp_x_in_signal = np.random.choice(all_x_in_signal,replace=False,
                size=n_patterns_per_signal)
            matrix_x_in_signal = np.vstack((matrix_x_in_signal, tmp_x_in_signal))
        matrix_x_in_signal = matrix_x_in_signal.astype("int32")

        random_pos = length_patterns * matrix_x_in_signal + matrix_x_in_segment
        # necessary : we cannot split without control for the #patterns
        # if array split does not result in an equal division
        modulo_split = idx_all_pos.size%(patterns[k].shape[0])
        if not modulo_split:
            idx_pos_patterns = np.array(np.split(idx_all_pos,
                patterns[k].shape[0]))
        else:
            idx_pos_patterns = np.array(np.split(idx_all_pos[:-modulo_split],
                patterns[k].shape[0]))

        for t in range(patterns[k].shape[0]):
            for i in range(samples_per_class):
                tmp_sample = k * samples_per_class + i
                for s in range(idx_pos_patterns.shape[1]):
                    start = (random_pos[i, idx_pos_patterns[t, s]])
                    # print(start)
                    X[tmp_sample, start:start+length_patterns] = patterns[k][t, :] * (mu + sigma * np.random.randn(1))
                y[i] = k

    # here we add the gaussian noise
    if noise>0. :
        X += noise * np.random.randn(n, p) # np.mean(1./p * np.linalg.norm(X, axis=-1))

    y = y.astype("int32")

    return X[:samples_per_class * n_classes, :], y[:samples_per_class * n_classes], patterns
