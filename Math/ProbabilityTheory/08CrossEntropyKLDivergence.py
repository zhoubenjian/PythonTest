'''
交叉熵

KL散度
'''
import numpy as np


def cross_entropy(p, q):
    """
    交叉熵
    :param p: 真实分布
    :param q: 预测分布
    :return: 交叉熵
    """
    return -np.sum(p * np.log(q + 1e-10))


# 真实：类别0
p_true = np.array([1.0, 0.0, 0.0])
