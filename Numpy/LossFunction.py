'''
常见随时函数（Loss Function）
'''
import numpy as np


'''
MSE（Mean Squared Error (MSE)）- 适用于回归问题（预测连续值，如房价、温度）

    均方误差计算的是所有样本的预测值与真实值之差的平方的平均值。
    
    公式：MSE = (1/n) * Σ(真实值ᵢ - 预测值ᵢ)²
        n：样本数量
        Σ：求和符号
        真实值ᵢ：第i个样本的真实值
        预测值ᵢ：模型对第 i 个样本的预测值
        
    特点：由于使用了平方，它对较大的误差惩罚更重（误差为 2 时，平方后贡献为 4；误差为 10 时，平方后贡献高达 100）。
'''
# 模拟数据
y_true = np.array([3, -.5, 2, 7, 4])
y_pred = np.array([2.5, 0, 2, 8, 5])

# 手动实现（MSE）
n = len(y_true)
squared_errors = (y_true - y_pred) ** 2     # 每个样本的误差平方
mse_manual = np.sum(squared_errors) / n     # 平均误差
print('手动计算MSE：%.2f' % mse_manual)


# 使用 sklearn 实现
from sklearn.metrics import mean_squared_error

mse_sklearn = mean_squared_error(y_true, y_pred)
print('Sklearn计算MSE（mean_squared_error）：%.2f' % mse_sklearn)



print('\n' + '-' * 50 + '\n')



'''
交叉熵损失（Cross-Entropy Loss） - 适用于分类问题（预测类别，如图片是猫还是狗）

    交叉熵衡量的是模型预测的概率分布与真实的概率分布之间的差异。在二分类中，真实分布通常是 [1, 0]（是类别 A）或 [0, 1]（是类别 B）。

    二分类公式（对数损失）：Log Loss = - (1/n) * Σ [真实值ᵢ * log(预测概率ᵢ) + (1 - 真实值ᵢ) * log(1 - 预测概率ᵢ)]

    直观理解：当真实标签为1时，我们希望模型预测的概率也接近 1。如果此时模型预测了一个很低的概率（比如 0.1），那么 log(0.1) 会是一个很大的负数，再乘以前面的负号，就会导致损失值变得很大，表示惩罚很重。
'''
# 模拟二分类示例：真实标签（1代表"是"，0代表"否"）
y_true_binary = np.array([1, 0, 0, 1])      # 真实类别：是，否，否，是

# 模型预测为"是"这个类别的概率
y_pred_prob = np.array([.9, .1, .2, .8])    # 预测概率：0.9, 0.1, 0.2, 0.8

# 手动实现 交叉熵损失（Cross-Entropy Loss）
sum = 0.0
for t, p in zip(y_true_binary, y_pred_prob):
    sum += ((t * np.log(p)) + ((1 - t) * np.log(1 - p)))
ce_loss_manual = -sum / len(y_true_binary)
print(f'手动计算交叉熵损失（Cross-Entropy Loss）：{ce_loss_manual}')


# 使用 sklearn 计算交叉熵损失（对数损失）
from sklearn.metrics import log_loss

ce_loss = log_loss(y_true_binary, y_pred_prob)
print(f'Sklearn计算交叉损失熵（log_loss）：{ce_loss}')



