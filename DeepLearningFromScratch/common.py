import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(a):
    return a
