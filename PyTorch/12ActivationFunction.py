'''
激活函数
    回归：无激活 / ReLU
        max(0,x)

    二分类：Sigmoid
        1/(1+e^-x)

    多分类：Softmax
        e^x / sum(e^x)
'''


import torch
import torch.nn as nn


relu = nn.ReLU()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)
