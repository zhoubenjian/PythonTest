import math


def entropy(p_list):
    """
    计算熵 H = -sum(p_i * log2(p_i))
    p_list: 各个类别的概率列表，如 [0.5, 0.5]
    :return:
    """
    ent = 0.0
    for p in p_list:
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def gini(p_list):
    """
    计算基尼指数 Gini = 1 - sum(p_i^2)
    p_list: 各个类别的概率列表
    :return:
    """
    g = 1.0
    for p in p_list:
        if p > 0:
            g -= p * p
    return g



