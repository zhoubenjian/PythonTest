'''
合页损失函数（Hinge Loss）

    从二分类开始（最直观）
        L(y, f(x)) = max(0, 1 - y * f(x))
        分类正确且自信（完全满意，不惩罚）           y * f(x) ≥ 1         0
        分类正确但不够自信（惩罚，要求拉大边界）      0 < y * f(x) < 1     1 - y * f(x)
        分类错误（重罚）                         y * f(x) ≤ 0         1 - y * f(x) ≥ 1

        y * f(x)叫做"函数间隔"（functional margin）
        y * f(x) > 0：分类正确。
        y * f(x) > 1：分类正确且超过了安全边界，损失为 0
        Hinge Loss 的" hinge（合页）"名字来源：损失函数图像像一个合页，在y * f(x) = 1处弯折


    多分类 Hinge Loss（Crammer-Singer 形式）
        L = 1/n * sum(max(0, scores[i, j] - correct_score + δ))
        scores[i, j]：错误类别的得分
        correct_score：正确类别的得分
        含义：错误类别得分比正确类别高多少？如果高出不够 1，就惩罚。

        正确类别的得分，应该比所有错误类别的得分都高出至少 1 个边界。
'''
import numpy as np


def manual_multi_hinge_loss(scores, y_true, δ=1.0):
    '''
    手动计算多分类 Hinge Loss
    :param scores: 每个样本的类别得分矩阵，形状为 (n_samples, n_classes)
    :param y_true: 真实标签向量，形状为 (n_samples,)
    :param δ: 损失函数的参数，用于控制惩罚的强度
    :return: 多分类 Hinge Loss 值
    '''
    # 样本数量
    n_samples = scores.shape[0]

    total_loss = 0.0
    for i in range(n_samples):
        correct_score = scores[i, y_true[i]]
        loss_i = 0.0
        for j in range(scores.shape[1]):
            if j == y_true[i]:
                continue
            loss_i += max(0, scores[i, j] - correct_score + δ)
        total_loss += loss_i
    return total_loss / n_samples


if __name__ == '__main__':

    '''
    计算Hinge Loss
    '''
    # 真实标签
    y_true = 1
    f_x = np.array([2.0, 1.0, 0.5, 0.0, -1.0])

    # 计算损失inge Loss
    hinge_loss = np.maximum(0, 1 - y_true * f_x)
    print('二分类 Hinge Loss:', hinge_loss, sep='')


    print('\n' + '=' * 50 + '\n')


    '''
    手动计算多分类 Hinge Loss
    '''
    # 真实标签[0:猫, 1:狗, 2:鸟]
    y_true = np.array([0, 1, 2])

    # 正确分类得分
    scores = np.array([
        [3.0, 1.0, 0.5],    # 样本1：猫，预测正确
        [1.0, 3.0, 0.5],    # 样本2：狗，预测正确
        [0.5, 0.5, 3.0],    # 样本3：鸟，预测正确
    ])

    right_loss = manual_multi_hinge_loss(scores, y_true)
    print(f"（手动计算）（正确分类）多分类 Hinge Loss: {right_loss:.4f}")        # 0.0000

    print('-' * 50)

    # 错误分类得分
    scores = np.array([
        [3.0, 1.0, 0.5],    # 样本1：猫，预测正确
        [3.0, 1.0, 0.5],    # 样本2：狗，但模型给"猫"打了最高分（错误！）
        [0.5, 0.5, 3.0],    # 样本3：鸟，预测正确
    ])

    wrong_loss = manual_multi_hinge_loss(scores, y_true)
    print(f"（手动计算）（错误分类）多分类 Hinge Loss: {wrong_loss:.4f}")        # 1.1667


    print('\n' + '=' * 50 + '\n')


    '''
    PyTorch 封装实现多分类 Hinge Loss
        loss(x, y) = sum(max(0, margin - x[y] + x[i])) / N
    '''
    import torch
    import torch.nn as nn

    # 输入：模型输出的原始得分（非概率）
    # p=1 对应标准 Hinge Loss，p=2 对应平方 Hinge Loss
    criterion = nn.MultiMarginLoss(p=1, margin=1.0)
    # 模拟数据：3个样本，4个类别
    scores = torch.tensor([
        [0.1, 0.2, 0.4, 0.8],   # 样本1：真实类别是3，预测正确
        [0.5, 0.3, 0.1, 0.1],   # 样本2：真实类别是0，预测正确
        [0.2, 0.6, 0.1, 0.1],   # 样本3：真实类别是1，预测正确
    ])
    # 真实标签
    y_true = torch.tensor([3, 0, 1])

    loss = criterion(scores, y_true)
    print(f"PyTorch MultiMarginLoss: {loss.item():.4f}")     # 0.4083


    print('\n' + '=' * 50 + '\n')


    '''
    TensorFlow / Keras 封装实现多分类 Hinge Loss
    '''
    import tensorflow as tf

    # 二分类 Hinge Loss
    # loss = maximum(0, 1 - y_true * y_pred)
    binary_hinge_loss = tf.keras.losses.Hinge()

    # ✅ 标签必须为 -1/1
    # 使用 Hinge Loss 时务必确保标签为 -1 或 +1，否则 y_true=0 的样本会被当作"负类"处理，导致 loss 被错误放大。
    y_true = [[-1., 1.], [-1., -1.]]
    y_pred = [[0.6, 0.4], [0.4, 0.6]]

    loss = binary_hinge_loss(y_true, y_pred)
    print(f"TensorFlow 二分类 Hinge Loss: {loss.numpy():.4f}")     # 1.3000

    print('-' * 50)

    # 多分类 Hinge Loss
    # 模拟数据（转换为独热编码）：2个样本，3个类别
    y_true = tf.keras.utils.to_categorical(np.array([0, 2]), num_classes=3)
    # 模型预测值（随机生成）
    y_pred = np.random.random((2, 3))

    loss = tf.keras.losses.categorical_hinge(y_true, y_pred)
    print(f"TensorFlow 多分类 Hinge Loss: {loss.numpy()}")


    print('\n' + '=' * 50 + '\n')


    '''
    Scikit-learn 封装实现多分类 Hinge Loss
    sklearn.metrics.hinge_loss 是评估指标，不是训练损失。它需要传入分类器的 decision_function 输出
    '''
    from sklearn import svm
    from sklearn.metrics import hinge_loss

    # 二分类 Hinge Loss
    # 训练SVM分类器
    X = [[0], [1]]
    y = [-1, 1]
    est = svm.LinearSVC()
    est.fit(X, y)

    # 获取决策值
    pred_decision = est.decision_function([[-2], [3], [0.5]])
    print(f'决策值：{pred_decision}')       # [-2.18181818  2.36363636  0.09090909]

    # 计算Hinge Loss
    loss = hinge_loss([-1, 1, 1], pred_decision)
    print(f'Scikit-learn 二分类 Hinge Loss: {loss:.4f}')     # 0.3030

    print('-' * 50)

    # 多分类 Hinge Loss
    # 训练多分类SVM分类器
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 1, 2, 3])
    est = svm.LinearSVC()
    est.fit(X, y)

    # 获取决策值
    pred_decision = est.decision_function([[-1], [2], [3]])
    y_true = [0, 2, 3]
    labels = np.array([0, 1, 2, 3])

    # 计算多分类 Hinge Loss（多分类时需要提供 labels 参数）
    loss = hinge_loss(y_true, pred_decision, labels=labels)
    print(f"Scikit-learn 多分类 Hinge Loss: {loss:.4f}")       # 0.5641

