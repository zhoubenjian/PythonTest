'''
Huber损失函数
    核心思想：
        误差小的时候，用 MSE（平滑，梯度稳定，收敛精确）
        误差大的时候，用 MAE（鲁棒，不被离群点带偏）

    适用场景：
        有离群点、需要鲁棒性
'''
import numpy as np
import torch
import torch.nn as nn


def huber_loss(y_pred, y_true, delta=1.0):
    '''
    手动实现Huber损失函数
    :param y_pred: 模型预测值
    :param y_true: 真实值
    :param delta: 阈值
    :return:
    '''
    is_small = np.abs(y_pred - y_true) <= delta
    squared_loss = 0.5 * (y_pred - y_true) ** 2
    linear_loss = delta * (np.abs(y_pred - y_true) - delta * 0.5)
    return np.mean(np.where(is_small, squared_loss, linear_loss))


if __name__ == '__main__':

    # 真实值
    y_pred = np.array([10, 20, 30])
    # 预测值
    y_true = np.array([11, 17, 35])
    delta = 1.0


    '''
    手动实现Huber损失函数值
    '''
    loss = huber_loss(y_pred, y_true, delta)
    print(f'手动Huber损失函数值: {loss:.4f}')


    print('\n' + '=' * 30 + '\n')


    '''
    PyTorch 封装实现Huber损失函数
    '''
    criterion = nn.HuberLoss()
    loss = criterion(torch.tensor(y_pred.astype(np.float32)), torch.tensor(y_true.astype(np.float32)))
    print(f'PyTorchHuber损失函数值: {loss.item():.4f}')

