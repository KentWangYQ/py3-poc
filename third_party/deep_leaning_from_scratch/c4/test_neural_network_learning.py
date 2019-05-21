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


class GradientTest(unittest.TestCase):
    def function_2(self, x):
        if x.ndim == 1:
            return np.sum(x ** 2)
        else:
            return np.sum(x ** 2, axis=1)

    def _numerical_gradient_no_batch(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)

        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)

            x[idx] = tmp_val - h
            fxh2 = f(x)

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp_val

        return grad

    def numerical_gradient(self, f, x):
        if x.ndim == 1:
            return self._numerical_gradient_no_batch(f, x)
        else:
            grad = np.zeros_like(x)

            for idx, x in enumerate(x):
                grad[idx] = self._numerical_gradient_no_batch(f, x)

            return grad

    def tangent_line(self, f, x):
        d = self.numerical_gradient(f, x)
        print(d)
        y = f(x) - d * x
        return lambda t: d * t + y

    def test_numerical_gradient(self):
        print(self.numerical_gradient(self.function_2, np.array([3.0, 4.0])))
        print(self.numerical_gradient(self.function_2, np.array([.0, 2.0])))
        print(self.numerical_gradient(self.function_2, np.array([3.0, .0])))

    def test_ng_2d(self):
        x0 = np.arange(-2, 2.5, .25)
        x1 = np.arange(-2, 2.5, .25)
        x, y = np.meshgrid(x0, x1)

        x = x.flatten()
        y = y.flatten()

        grad = self.numerical_gradient(self.function_2, np.array([x, y]))

        plt.figure()
        plt.quiver(x, y, -grad[0], -grad[1], angles='xy', color='#666666')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.grid()
        plt.legend()
        plt.draw()
        plt.show()

    def gradient_descent(self, f, init_x, lr=.01, step_num=100):
        """梯度下降法"""
        x = init_x
        x_history = []

        for i in range(step_num):
            x_history.append(x.copy())

            grad = self.numerical_gradient(f, x)
            x -= lr * grad

        return x, np.array(x_history)

    def test_gradient_descent(self):
        init_x = np.array([-3.0, 4.0])

        lr = 0.1
        step_num = 20
        x, x_history = self.gradient_descent(self.function_2, init_x, lr=lr, step_num=step_num)

        plt.plot([-5, 5], [0, 0], '--b')
        plt.plot([0, 0], [-5, 5], '--b')
        plt.plot(x_history[:, 0], x_history[:, 1], 'o')

        plt.xlim(-3.5, 3.5)
        plt.ylim(-4.5, 4.5)
        plt.xlabel('X0')
        plt.ylabel('X1')
        plt.show()
