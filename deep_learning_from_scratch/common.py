import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(a):
    return a


def softmax(a):
    """
    softmax 输出层的激活函数，一般用于分类问题
    输出结果总和为1，体现为概率。
    :param a:
    :return:
    """
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
