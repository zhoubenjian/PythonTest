'''
二元分类交叉熵（Binary Cross-Entropy Loss）：
    用于二分类任务，将输入映射到0到1之间的概率值，用于计算分类错误的损失；

    损失函数的定义为：Loss = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]

    直观理解:
        如果真实标签 y=1：损失为 -log(y_pred)，预测概率越接近 1 损失越小
        如果真实标签 y=0：损失为 -log(1 - y_pred)，预测概率越接近 0 损失越小
'''
import torch
import torch.nn as nn
import numpy as np


def binary_cross_entropy(y_true, y_pred, epsilon=1e-7):
    """
    手动实现二元交叉损失函数
    :param y_true: 真实标签，0或1
    :param y_pred: 预测概率，0到1之间的浮点数
    :param epsilon: 防止对0或1取对数，设置一个小的常量（避免梯度爆炸，1e-7 是业界通用经验值）
    :return: 二元交叉损失值
    """

    '''
    1.稳定数值
    '''
    # 预测值限制在[epsilon, 1-epsilon]
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

    '''
    2.计算每个样本损失
    '''
    # y_true = 1: loss = -log(y_pred)
    # y_true = 0: loss = -log(1 - y_pred)
    sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    '''
    3.计算平均损失
    '''
    return np.mean(sample_losses)


if __name__ == '__main__':

    # 真实标签
    y_true = np.array([1, 0, 1, 0, 1])


    '''
    PyTorch损失函数 
    '''
    # 自带Sigmoid + 二元交叉熵
    criterion = nn.BCEWithLogitsLoss()
    # 模型原始输出（Logits，未经过 Sigmoid）
    # 注意：这里是任意实数，不是概率！
    logits = torch.tensor([2.3, -1.5, 0.8, -2.1, 3.2])
    loss = criterion(logits, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))
    print(f'PyTorch损失函数的损失值(Sigmoid + 二元交叉熵): {loss.item():.4f}')    # 0.1647

    print("-" * 30)

    # 二元交叉熵(已有概率值)
    criterion = nn.BCELoss()
    loss = criterion(torch.tensor([0.9, 0.2, 0.7, 0.1, 0.99]), torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))
    print(f'PyTorch损失函数的损失值(二元交叉熵): {loss.item():.4f}')    # 0.1601


    print("\n" + "=" * 50 + "\n")


    '''
    一般情况
    '''
    # 模型预测概率
    y_pred = np.array([0.9, 0.2, 0.7, 0.1, 0.99])
    loss = binary_cross_entropy(y_true, y_pred)
    print(f'(手动实现)一般情况的损失值: {loss:.4f}')          # 0.1601

    print('-' * 30)

    '''
    特殊情况(完全正确)
    '''
    y_pred_perfect = np.array([0.99, 0.01, 0.99, 0.01, 0.99])
    loss_perfect = binary_cross_entropy(y_true, y_pred_perfect)
    print(f'(手动实现)完全正确的损失值: {loss_perfect:.4f}')  # 0.0101

    print('-' * 30)

    '''
    特殊情况(完全错误)
    '''
    y_pred_infect = np.array([0.01, 0.99, 0.01, 0.99, 0.01])
    loss_infect = binary_cross_entropy(y_true, y_pred_infect)
    print(f'(手动实现)完全错误的损失值: {loss_infect:.4f}')   # 4.6052








