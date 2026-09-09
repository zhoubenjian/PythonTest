'''
多分类损失函数

    为什么手动实现多分类交叉熵损失函数时，y_true 是 one-hot 编码（二维矩阵）？PyTorch 为什么需要 one-hot 编码（一维向量）？
        因为手动实现时，我们是在做"矩阵逐元素运算"（Element-wise），必须用 One-hot 来筛选出真实类别的概率；
        而 PyTorch 内部用"索引取数"（Indexing）实现了同样的效果，所以不需要 One-hot。
'''
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf


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

    '''
    手动实现多分类交叉熵损失函数
    '''
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
    print(f'（手动实现）多分类交叉熵损失值: {loss:.4f}')       # 0.4459

    # 计算每个样本损失
    sample_losses = np.sum(y_true * np.log(y_pred + 1e-7), axis=1)
    print(f'每个样本损失: {sample_losses}')       # [-0.3566748  -0.51082546 -0.69314698 -0.22314343]


    print('\n' + '=' * 30 + '\n')


    '''
    PyTorch 封装实现多分类交叉熵损失函数
    '''
    # 自带 Softmax + 交叉熵
    criterion = nn.CrossEntropyLoss()

    # 真实标签（非独热编码）
    # 类别0（猫），类别1（狗），类别2（鸟），类别3（鱼）
    y_true = torch.tensor([0, 1, 2, 3])

    # 模型输出：Logits（未经过 Softmax）
    # shape: (batch_size, num_classes) = (4, 4)
    # 原始分数（未归一化）
    logits = torch.tensor([
        [0.1, 0.7, 0.15, 0.05],  # 样本1：最高分 0.7 在索引1（狗）
        [0.6, 0.2, 0.15, 0.05],  # 样本2：最高分 0.6 在索引0（猫）
        [0.1, 0.2, 0.5, 0.2],    # 样本3：最高分 0.5 在索引2（鸟）
        [0.05, 0.1, 0.05, 0.8],  # 样本4：最高分 0.8 在索引3（鱼）
    ], dtype=torch.float32)

    loss = criterion(logits, y_true)
    print(f"PyTorch多分类交叉熵损失值: {loss.item():.4f}")


    print('\n' + '=' * 30 + '\n')


    '''
    TensorFlow 封装实现多分类交叉熵损失函数
    '''
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 真实标签（非独热编码）
    # 类别0（猫），类别1（狗），类别2（鸟），类别3（鱼）
    y_true = tf.constant([0, 1, 2, 3])

    # 模型输出：Logits（未经过 Softmax）
    # shape: (batch_size, num_classes) = (4, 4)
    # 原始分数（未归一化）
    logits = tf.constant([
        [0.1, 0.7, 0.15, 0.05],     # 样本1：最高分 0.7 在索引1（狗），预测：猫  ❌️
        [0.6, 0.2, 0.15, 0.05],     # 样本2：最高分 0.6 在索引0（猫），预测：狗  ❌️
        [0.1, 0.2, 0.5, 0.2],       # 样本3：最高分 0.5 在索引2（鸟），预测：鸟  ✅️
        [0.05, 0.1, 0.05, 0.8],     # 样本4：最高分 0.8 在索引3（鱼），预测：鱼  ✅️
    ], dtype=tf.float32)

    loss = loss_fn(y_true, logits)
    print(f"TensorFlow多分类交叉熵损失值: {loss.numpy():.4f}")






