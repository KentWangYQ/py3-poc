import unittest
from deep_learning_from_scratch import mnist


class MNISTTest(unittest.TestCase):
    def test_test(self):
        mn = mnist.MNIST()
        mn.test()
