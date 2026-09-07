'''
多分类损失函数
'''
import numpy as np


def multi_cross_entropy(y_true, y_pred, epsilon=1e-7):
    '''
    多分类交叉熵损失函数手动实现
    :param y_true: 真实标签（one-hot编码）
    :param y_pred: 预测概率（softmax后的概率值）
    :param epsilon: 防止log(0)错误的常量
    :return: 多分类交叉熵损失值
    '''
    # 确保预测概率稳定在 (epsilon, 1 - epsilon) 范围内，避免 log(0) 错误
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 计算每个样本损失
    sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)

    # 返回平均损失
    return np.mean(sample_losses)


if __name__ == '__main__':

    # 4个真实样本（独热编码），4个类别（狗，猫，鸟，鱼）
    y_true = np.array([
        [0, 1, 0, 0],   # 样本1：猫
        [1, 0, 0, 0],   # 样本2：狗
        [0, 0, 1, 0],   # 样本3：鸟
        [0, 0, 0, 1],   # 样本4：鱼
    ])

    # 预测概率（softmax后的概率值）
    y_pred = np.array([
        [0.1, 0.7, 0.15, 0.05],  # 样本1预测：猫（概率0.7）
        [0.6, 0.2, 0.15, 0.05],  # 样本2预测：狗（概率0.6）
        [0.1, 0.2, 0.5, 0.2],    # 样本3预测：鸟（概率0.5）
        [0.05, 0.1, 0.05, 0.8],  # 样本4预测：鱼（概率0.8）
    ])

    loss = multi_cross_entropy(y_true, y_pred)
    print(f'（手动实现）多分类交叉熵损失值: {loss:.4f}')

    # 计算每个样本损失
    sample_losses = np.sum(y_true * np.log(y_pred + 1e-7), axis=1)
    print(f'每个样本损失: {sample_losses}')



