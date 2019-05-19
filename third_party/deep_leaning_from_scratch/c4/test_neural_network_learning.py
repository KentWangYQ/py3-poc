import unittest
import numpy as np


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
