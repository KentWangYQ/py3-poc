import unittest
import numpy as np


class PerceptronTest(unittest.TestCase):
    def AND(self, x1, x2):
        w1, w2, theta = .5, .5, .7
        tmp = x1 * w1 + x2 * w2
        if tmp <= theta:
            return 0
        elif tmp > theta:
            return 1

    def AND2(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([.5, .5])
        b = -.7
        tmp = np.sum(x * w) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1

    def OR(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([.5, .5])
        b = -.1
        tmp = np.sum(x * w) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1

    def NAND(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([-.5, -.5])
        b = .7
        tmp = np.sum(x * w) + b
        if tmp <= 0:
            return 0
        elif tmp > 0:
            return 1

    def XOR(self, x1, x2):
        s1 = self.OR(x1, x2)
        s2 = self.NAND(x1, x2)
        return self.AND(s1, s2)

    def test_gate(self):
        self.assertEqual(0, self.AND(0, 0))
        self.assertEqual(0, self.AND(0, 1))
        self.assertEqual(0, self.AND(1, 0))
        self.assertEqual(1, self.AND(1, 1))

        self.assertEqual(0, self.AND2(0, 0))
        self.assertEqual(0, self.AND2(0, 1))
        self.assertEqual(0, self.AND2(1, 0))
        self.assertEqual(1, self.AND2(1, 1))

        self.assertEqual(0, self.OR(0, 0))
        self.assertEqual(1, self.OR(0, 1))
        self.assertEqual(1, self.OR(1, 0))
        self.assertEqual(1, self.OR(1, 1))

        self.assertEqual(1, self.NAND(0, 0))
        self.assertEqual(1, self.NAND(0, 1))
        self.assertEqual(1, self.NAND(1, 0))
        self.assertEqual(0, self.NAND(1, 1))

        self.assertEqual(0, self.XOR(0, 0))
        self.assertEqual(1, self.XOR(0, 1))
        self.assertEqual(1, self.XOR(1, 0))
        self.assertEqual(0, self.XOR(1, 1))

    def test_wb(self):
        x = np.array([0, 1])
        w = np.array([.5, .5])
        b = -.7
        print(w * x)
        print(np.sum(w * x))
        print(np.sum(w * x) + b)
