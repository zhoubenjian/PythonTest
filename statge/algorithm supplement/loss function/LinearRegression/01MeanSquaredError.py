'''
均方误差损失函数(Mean of Squared Error)L2 Loss
    	核心特点：
    	    对大误差惩罚极重

    	适用场景：
    	    误差分布接近高斯、无离群点
'''
import numpy as np
import torch
import torch.nn as nn


def manual_mse(y_true, y_pred):
    '''
    手动实现计算均方误差损失值
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 均方误差损失值
    '''
    return np.mean((y_true - y_pred) ** 2)


if __name__ == '__main__':

    # 真实值
    y_true = np.array([10, 10, 10])
    # 预测值
    y_pred = np.array([9, 12, 7])


    '''
    手动计算均方误差损失值
    '''
    mse = manual_mse(y_true, y_pred)
    print(f"手动实现计算均方误差损失值: {mse:.4f}")          # 4.6667


    print('\n' + '=' * 30 + '\n')


    '''
    PyTorch 封装实现均方误差损失函数
    '''
    criterion = nn.MSELoss()
    loss = criterion(torch.tensor(y_true.astype(np.float32)), torch.tensor(y_pred.astype(np.float32)))
    print(f"PyTorch均方误差损失值: {loss.item():.4f}")     # 4.6667

