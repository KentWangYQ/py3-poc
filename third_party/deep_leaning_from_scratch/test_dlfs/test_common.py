import unittest
from deep_learning_from_scratch.common import *


class CommonTest(unittest.TestCase):
    def test_numerical_gradient(self):
        print(numerical_gradient(function_2, np.array([3.0, 4.0])))

        x0 = np.arange(-2, 2.5, .25)
        x1 = np.arange(-2, 2.5, .25)
        x, y = np.meshgrid(x0, x1)

        x = x.flatten()
        y = y.flatten()

        print(numerical_gradient(function_2, np.array([x, y])))
