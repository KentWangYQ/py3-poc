import unittest
from DeepLearningFromScratch import neural_network


class ThreeLayerNeuralNetworkTest(unittest.TestCase):
    def test_run(self):
        tlnn = neural_network.ThreeLayerNeuralNetwork()
        tlnn.run()
