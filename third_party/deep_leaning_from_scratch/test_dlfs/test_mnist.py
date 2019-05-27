import unittest
from deep_learning_from_scratch import mnist


class MNISTTest(unittest.TestCase):
    def test_test(self):
        mn = mnist.MNIST()
        mn.test()

    def test_show_imgs(self):
        mn = mnist.MNIST()
        (x, t), _ = mn.load_data(normalize=False, flatten=True)
        mn.show_imgs(x[0:1200])
