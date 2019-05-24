import numpy as np
from .common import *


class ThreeLayerNeuralNetwork:
    def init_network(self):
        """
        初始化神经网络
        包括权重和重置
        按照神经网络的实现惯例，只把权重记为大写字母W，其他的(偏置或中间结果等)都用小写字母表示。
        :return:
        """
        network = {
            'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
            'b1': np.array([0.1, 0.2, 0.3]),
            'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
            'b2': np.array([0.1, 0.2]),
            'W3': np.array([[0.1, 0.3], [0.2, 0.4]]),
            'b3': np.array([0.1, 0.2]),
        }

        return network

    def forward(self, network, x):
        """
        前向处理
        封装了将输入信号转换为输出信号的处理过程
        前向表示从输入到输出方向的传递处理
        :param network:
        :param x:
        :return:
        """
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = identity_function(a3)

        return y

    def run(self):
        network = self.init_network()
        x = np.array([1.0, 0.5])
        y = self.forward(network, x)
        print(y)
