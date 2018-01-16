import numpy as np
from scipy import signal


def product(f, g):
    def eval_prod(x):
        return f(x) * g(x)
    return eval_prod

class SignalGenerator1D():
    """class for the generation of one dimensional signals for classification
    tasks. We will generate samples belonging to different classes, which number
    is fixed when the SignalGenerator1D object is defined. """
    def __init__(self, samples, length, classes=8):
        if classes < 2 or samples < classes or length < 2:
            raise ValueError("check again the initialization values")
        self.samples = samples
        self.length = length
        self.classes = classes
        self.pattern_length_array = None
        self.pattern_amplitude_array = None
        self.function_list = [signal.tukey, signal.boxcar, signal.gaussian,
            signal.triang]
        self.function_list += [product(func, np.sin) for f in self.function_list]


    def generate_signal(self, sparsity, noise=0., balanced=True):
        print(self.function_list[1](100))
        return self


def main():
    samples=1000
    sg = SignalGenerator1D(samples=samples, length=1000, classes=8)
    sg.generate_signal(0)


if __name__ == '__main__':
    main()
