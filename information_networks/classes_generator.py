import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def product(f):
    """ product with sin function"""
    def eval_prod(n):
        """ n : support dimension"""
        return f(n) * np.sin(4 * np.pi * np.arange(n) / n)
    return eval_prod

def gaussian(std):
    def eval_gaussian(n):
        return signal.gaussian(n, std)
    return eval_gaussian


class SignalGenerator1D():
    """class for the generation of one dimensional signals for classification
    tasks. We will generate samples belonging to eight classes, which number
    is fixed when the SignalGenerator1D object is defined. """
    def __init__(self, samples, length):
        if length < 2 or samples < 2:
            raise ValueError("check again the initialization values")
        self.classes = 8
        self.samples = samples
        self.length = length
        self.pattern_length_array = None
        self.pattern_amplitude_array = None
        self.function_list = [signal.tukey, signal.boxcar, gaussian(std=10.),
            signal.triang]
        self.function_list += [product(func) for func in self.function_list]
        self.X = np.zeros((self.samples, self.length))
        self.y = np.zeros(self.samples)

    def generate_signal(self, sparsity, noise=0., balanced=True):

        sigma_supp = 1e-1
        sigma_ampl = 1e-1

        segments = 10  # ten segments
        length_patterns = self.length / segments
        self.pattern_length_array = np.random.uniform(low=int(length_patterns*0.8), high=length_patterns, size=self.classes)
        self.pattern_amplitude_array = np.random.uniform(low=1, high=5*self.classes, size=self.classes)

        samples_per_class = self.samples / self.classes

        all_x_in_signal = np.arange(0, segments-1, 2)  # step=2 to avoid overlap
        all_x_in_segment = np.arange(length_patterns)  # pos in segment

        n_patterns_per_signal = int((1-sparsity) * all_x_in_signal.size)
        if n_patterns_per_signal < 1:
            raise ValueError("sparsity value is too high")
        idx_all_pos = np.arange(n_patterns_per_signal)

        for idx_f, f in enumerate(self.function_list):
            matrix_x_in_segment = np.random.choice(all_x_in_segment,
                size=(samples_per_class, n_patterns_per_signal))

            matrix_x_in_signal = np.array([]).reshape(0, n_patterns_per_signal)
            for j in range(samples_per_class):
                tmp_x_in_signal = np.random.choice(all_x_in_signal,replace=False,
                    size=n_patterns_per_signal)
                matrix_x_in_signal = np.vstack((matrix_x_in_signal, tmp_x_in_signal))
            matrix_x_in_signal = matrix_x_in_signal.astype("int32")
            random_pos = self.length/segments * matrix_x_in_signal + matrix_x_in_segment

            # modulo_split = idx_all_pos.size % self.templates_per_class
            # if not modulo_split:
            #     idx_pos_patterns = np.array(np.split(idx_all_pos, 1))
            # else:
            #     idx_pos_patterns = np.array(np.split(idx_all_pos[:-modulo_split],
            #         1))
            for i in range(samples_per_class):
                tmp_sample = idx_f * samples_per_class + i
                for t in range(n_patterns_per_signal):
                    start = (random_pos[i, idx_all_pos[t]])
                    blur_support = 1 + sigma_supp * np.abs(np.random.randn())
                    support = (blur_support * self.pattern_length_array[idx_f]).astype("int")
                    if support + start > self.length:
                        support = self.length - start
                    tmp_pattern = self.pattern_amplitude_array[idx_f] * f(support)
                    self.X[tmp_sample, start:start+support] = tmp_pattern * (1 + sigma_ampl * np.random.randn())
            self.y[idx_f * samples_per_class: (idx_f+1) * samples_per_class] = idx_f

        return self

        # print(self.function_list[1](100))
        # return self


def main():
    samples=1000
    sg = SignalGenerator1D(samples=samples, length=1000)
    sg.generate_signal(0)
    X = sg.X
    for i in np.arange(0, 1000, 125):
        plt.plot(X[i, :])
    plt.show()

if __name__ == '__main__':
    main()
