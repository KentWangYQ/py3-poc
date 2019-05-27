import numpy as np


# region 激活函数(activation function)
def sigmoid(x):
    """
    sigmoid函数很早就开始作为激活函数使用
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    ReLU函数(Rectified Linear Unit)
    最近主要使用该函数作为激活函数
    :param x:
    :return:
    """
    return np.maximum(0, x)


# endregion

# region 输出层函数
def identity_function(a):
    """
    恒等函数
    一般用于回归问题
    :param a:
    :return:
    """
    return a


def softmax(a):
    """
    softmax函数
    一般用于分类问题
    输出结果总和为1，体现为概率。
    :param a:
    :return:
    """
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# endregion

# region 损失函数(loss function)
def mean_squared_error(y, t):
    """
    均方误差
    :param y: 神经网络的输出
    :param t: 监督数据(one-hot)
    :return:
    """
    return 0.5 * np.sum((y - t) ** 2)


def mean_squared_error_mini_batch(y, t):
    """
    均方误差的mini-batch版
    :param y: 神经网络的输出
    :param t: 监督数据(one-hot)
    :return:
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size


def cross_entropy_error(y, t):
    """
    交叉熵误差
    np.log(0)会变为负无穷大-inf，导致后续计算无法进行，
    引入微小值delta，作为保护策略。
    :param y: 神经网络的输出
    :param t: 监督数据(one-hot)
    :return:
    """
    delta = 1e-7  # 保护策略
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_mini_batch(y, t):
    """
    交叉熵误差的mini-batch版
    :param y: 神经网络的输出
    :param t: 监督数据(one-hot)
    :return:
    """
    delta = 1e-7
    if y.ndim == 1:
        # 如果y为一维，即只有一条测试数据
        # 将y和t转换成1×size的矩阵，适配下面的计算
        t = t.reshape(1, t.size)
        y = y.reshare(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size


# endregion


# region 数值微分(numerical differentiation)
def numerical_diff(f, x):
    """
    数值微分
    :param f:
    :param x:
    :return:
    """
    h = 1e-4  # 不能使用太小的数值，否则会产生舍入误差(rounding error)
    return (f(x + h) - f(x - h)) / (2 * h)
# endregion
