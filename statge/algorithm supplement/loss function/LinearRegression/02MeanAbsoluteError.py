'''
均绝对误差损失函数(Mean of Absolute Error)L1 Loss
    	核心特点：
    	    对所有误差一视同仁

    	适用场景：
    	    有离群点、需要鲁棒性
'''
import numpy as np
import torch
import torch.nn as nn


def manual_mae(y_true, y_pred):
    '''
    手动实现计算均绝对误差损失值
    :param y_true: 真实标签
    :param y_pred: 模型预测
    :return: 均绝对误差损失值
    '''
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


if __name__ == '__main__':

    # 真实值
    y_true = np.array([10, 10, 10])
    # 预测值
    y_pred = np.array([9, 12, 7])

    '''
    手动计算均绝对误差损失值
    '''
    mae = manual_mae(y_true, y_pred)
    print(f"手动实现计算均绝对误差损失值: {mae:.4f}")         # 2.0000


    print('\n' + '=' * 30 + '\n')


    '''
    PyTorch 封装实现均绝对误差损失函数
    '''
    criterion = nn.L1Loss()
    loss = criterion(torch.tensor(y_pred.astype(np.float32)), torch.tensor(y_true.astype(np.float32)))
    print(f"PyTorch均绝对误差损失值: {loss.item():.4f}")    # 2.0000
