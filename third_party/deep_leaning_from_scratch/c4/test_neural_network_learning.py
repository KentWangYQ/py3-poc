import unittest
import numpy as np
from matplotlib.pylab import plt
from mpl_toolkits.mplot3d import Axes3D as _3d


def mean_squared_error(y, t):
    """均方误差(mean squared error)"""
    return .5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """交叉熵误差(cross entropy error)"""
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


class LearningTest(unittest.TestCase):
    def test_mean_squared_error(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [.1, .05, .6, .0, .05, .1, .0, .1, .0, .0]
        print(mean_squared_error(np.array(y), np.array(t)))

        y = [.1, .05, .1, .0, .05, .1, .0, .6, 0, .0]
        print(mean_squared_error(np.array(y), np.array(t)))

    def test_cross_entropy_error(self):
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y = [.1, .05, .6, .0, .05, .1, .0, .1, .0, .0]
        print(cross_entropy_error(np.array(y), np.array(t)))

        y = [.1, .05, .1, .0, .05, .1, .0, .6, 0, .0]
        print(cross_entropy_error(np.array(y), np.array(t)))


class NumericalDifferentiationTest(unittest.TestCase):
    def numerical_diff(self, f, x):
        h = 1e-4
        return (f(x + h) - f(x - h)) / (2 * h)

    def function_1(self, x):
        return .01 * x ** 2 + .1 * x

    def test_numerical_diff(self):
        x = np.arange(.0, 20.0, .1)
        y = self.function_1(x)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(x, y)
        plt.show()

        print(self.numerical_diff(self.function_1, 5))
        print(self.numerical_diff(self.function_1, 10))

    def function_2(self, x, y):
        return x ** 2 + y ** 2

    def test_function_2(self):
        fig = plt.figure()
        ax = _3d(fig)
        x = np.arange(-3.0, 3.0, .1)
        y = np.arange(-3.0, 3.0, .1)
        x, y = np.meshgrid(x, y)
        z = self.function_2(x, y)
        ax.plot_surface(x, y, z,
                        rstride=1,
                        cstride=1,
                        cmap='rainbow'
                        )
        plt.show()

    def function_tmp1(self, x0):
        return x0 * x0 + 4.0 ** 2.0

    def function_tmp2(self, x1):
        return 3.0 ** 2.0 + x1 * x1

    def test_tmp(self):
        x, y = 3.0, 4.0
        print(self.numerical_diff(self.function_tmp1, x))
        print(self.numerical_diff(self.function_tmp2, y))
