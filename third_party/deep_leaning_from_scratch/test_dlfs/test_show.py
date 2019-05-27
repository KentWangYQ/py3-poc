import unittest
from deep_learning_from_scratch.show import *


class ShowTest(unittest.TestCase):
    def test_function_2_show(self):
        function_2_show(np.array([np.arange(-3.0, 3, 0.1), np.arange(-3.0, 3.0, 0.1)]))

    def test_numerical_gradient_show(self):
        numerical_gradient_show(x=(np.arange(-3.0, 3.1, 0.5), np.arange(-3.0, 3.1, 0.5)),
                                f=function_2)
