from .dataset.mnist import *
from .common import *
from .neural_network import ThreeLayerNeuralNetwork


class MNIST:
    def init_network(self):
        with open(dataset_dir + '/sample_weight.pkl', 'rb') as f:
            network = pickle.load(f)
        return network

    def test(self):
        """
        测试训练结果
        :return:
        """
        # 加载测试数据(测试图片, 测试标签)
        (x, t), _ = load_mnist(normalize=True, flatten=True, one_hot_label=False)
        # 加载训练结果神经网络
        network = self.init_network()
        # 三层神经网络
        tlnn = ThreeLayerNeuralNetwork()

        batch_size = 100  # 批数量
        accuracy_cnt = 0
        # 批量执行测试
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = tlnn.forward(network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == t[i:i + batch_size])

        print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
