import unittest
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


class NeuralNetworkTest(unittest.TestCase):
    def step_function(self, x):
        return np.array(x > 0, dtype=np.int)

    def test_step_function(self):
        x = np.arange(-5.0, 5.0, .1)
        y = self.step_function(x)
        plt.plot(x, y)
        plt.ylim(-.1, 1.1)
        plt.show()

    def test_sigmoid(self):
        x = np.arange(-5.0, 5.0, .1)
        y = sigmoid(x)
        print(y)
        plt.plot(x, y)
        plt.ylim(-.1, 1.1)
        plt.show()

    def relu(self, x):
        return np.maximum(0, x)

    def test_relu(self):
        x = np.arange(-5.0, 5.0, .1)
        y = self.relu(x)
        plt.plot(x, y)
        plt.ylim(-1)
        plt.show()


class MultidimensionalArrayTest(unittest.TestCase):
    def test_basic(self):
        a = np.array([1, 2, 3, 4])
        print(a)
        print(np.ndim(a))
        print(a.shape)
        print(a.shape[0])

        b = np.array([[1, 2], [3, 4], [5, 6]])
        print(b)
        print(np.ndim(b))
        print(b.shape)

        a = np.array([[1, 2], [3, 4]])
        print(a.shape)
        b = np.array([[5, 6], [7, 8]])
        print(a.shape)
        print(np.dot(a, b))

        a = np.array([[1, 2], [3, 4], [5, 6]])
        print(a.shape)
        b = np.array([[1, 2, 3], [4, 5, 6]])
        print(b.shape)
        print(np.dot(a, b))

    def test_nn_inner_product(self):
        x = np.array([1, 2])
        print(x.shape)
        w = np.array([[1, 3, 5], [2, 4, 6]])
        print(w.shape)
        y = np.dot(x, w)
        print(y)

    def init_network(self):
        network = {}
        network['w1'] = np.array([[.1, .3, .5], [.2, .4, .6]])
        network['b1'] = np.array([.1, .2, .3])
        network['w2'] = np.array([[.1, .4], [.2, .5], [.3, .6]])
        network['b2'] = np.array([.1, .2])
        network['w3'] = np.array([[.1, .3], [.2, .4]])
        network['b3'] = np.array([.1, .2])
        return network

    def identity_function(self, a):
        return a

    def forward(self, network, x):
        w1, w2, w3 = network['w1'], network['w2'], network['w3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = self.identity_function(a3)

        return y

    def test_multi_network(self):
        network = self.init_network()
        x = np.array([1.0, .5])
        y = self.forward(network, x)
        print(y)

    def softmax(self, a):
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

    def test_softmax(self):
        a = np.array([.3, 2.9, 4.0])
        y = self.softmax(a)
        print(y)

    def test_softmax_out_of_memory(self):
        a = np.array([1010, 1000, 990])
        y1 = self.softmax(a)
        print(y1)

        c = np.max(a)
        a2 = a - c
        print(a2)
        y2 = self.softmax(a2)
        print(y2)

    def softmax_improve(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

    def test_softmax_improve(self):
        a = np.array([1010, 1000, 990])
        print(self.softmax_improve(a))


from third_party.deep_leaning_from_scratch.dataset.mnist import load_mnist
from PIL import Image
import pickle


class MNISTTest(unittest.TestCase):
    def test_load_mnist(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        print(x_train.shape)
        print(t_train.shape)
        print(x_test.shape)
        print(t_test.shape)

    def test_show_img(self):
        def img_show(img):
            pil_img = Image.fromarray(np.uint8(img))
            pil_img.show()

        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        img = x_train[0]
        label = t_train[0]
        print(label)

        print(img.shape)
        img = img.reshape(28, 28)
        print(img.shape)

        img_show(img)

    def get_data(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_test, t_test

    def init_network(self):
        with open('sample_weight.pkl', 'rb') as f:
            network = pickle.load(f)

        return network

    def predict(self, network, x):
        w1, w2, w3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = softmax(a3)

        return y

    def test_mnist_check(self):
        x, t = self.get_data()
        network = self.init_network()

        accuracy_cnt = 0
        for i in range(len(x)):
            y = self.predict(network, x[i])
            p = np.argmax(y)
            if p == t[i]:
                accuracy_cnt += 1

        print('Accuracy:' + str(float(accuracy_cnt) / len(x)))

    def test_batch_mnist_check(self):
        x, t = self.get_data()
        network = self.init_network()

        batch_size = 100
        accuracy_cnt = 0
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = self.predict(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == t[i:i + batch_size])
        print('Accuracy:' + str(float(accuracy_cnt) / len(x)))
