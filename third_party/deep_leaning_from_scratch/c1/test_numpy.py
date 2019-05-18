import unittest
import numpy as np


class NunPyTest(unittest.TestCase):
    def test_first(self):
        x = np.array([1.0, 2.0, 3.0])
        print(x)
        print(type(x))

        y = np.array([2.0, 4.0, 6.0])
        print(x + y)
        print(x - y)
        print(x * y)
        print(x / y)

        print(x / 2.0)
        print('====================')

        a = np.array([[1, 2], [3, 4]])
        print(a)
        print(a.shape)
        print(a.dtype)

        b = np.array([[3, 0], [0, 6]])
        print(a + b)
        print(a * b)

        print(a * 10)
        print('====================')

        a = np.array([[1, 2], [3, 4]])
        b = np.array([10, 20])
        print(a * b)
        print('====================')

        x = np.array([[51, 55], [14, 19], [0, 4]])
        print(x)
        print(x[0])
        print(x[0][1])

        for row in x:
            print(row)

        x = x.flatten()
        print(x)
        print(x[np.array([0, 2, 4])])
        print(x > 15)
        print(x[x > 15])
