import unittest
import numpy as np
from ..common.functions import softmax, cross_entropy_error
from ..common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


class SimpleNetTest(unittest.TestCase):
    def test_simple_net(self):
        net = simpleNet()
        print(net.W)

        x = np.array([.6, .9])
        p = net.predict(x)
        print(p)

        t = np.array([0, 0, 1])  # 正确解标签
        print(net.loss(x, t))

        dw = numerical_gradient(lambda w: net.loss(x, t), net.W)
        print(dw)
