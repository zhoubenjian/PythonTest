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
import matplotlib.pyplot as plt
import numpy as np

# 导入中文字体配置
import common.mpl_config


'''
可视化 Hinge Loss 损失函数
'''
# 函数间隔[-3, 3]
margin = np.linspace(-3, 3, 500)
hinge_loss = np.maximum(0, 1 - margin)

plt.figure(figsize=(10, 6))
plt.plot(margin, hinge_loss, linewidth=2.5, color='red', label='Hinge Loss')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='分类边界（margin=0）')
plt.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='安全边界（margin=1）')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)

plt.xlabel('函数间隔 (y * f(x))', fontsize=12)
plt.ylabel('损失值', fontsize=12)

plt.title('Hinge Loss损失函数', fontsize=14)
plt.legend(fontsize=11)

plt.grid(True, alpha=0.3)
plt.show()


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
    print('Hinge Loss:', hinge_loss, sep='')


    print('\n' + '=' * 50 + '\n')


    scores = np.array([
        [3.0, 1.0, 0.5],
        [1.0, 3.0, 0.5],
        [0.5, 0.5, 3.0],
    ])

    y_true = np.array([0, 1, 2])

    loss = manual_multi_hinge_loss(scores, y_true)
    print(f"多分类 Hinge Loss: {loss:.4f}")


