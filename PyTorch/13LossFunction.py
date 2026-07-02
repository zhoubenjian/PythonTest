'''
损失函数
    均方误差损失函数（MSE Loss）
        用于回归任务，计算预测值与真实值之间的均方误差。
        用于优化模型的权重，使预测值更真实。

    交叉熵损失函数（Cross-Entropy Loss）
        用于多分类任务，计算预测值与真实值之间的交叉熵损失。
        用于优化模型的权重，使预测值更真实。
'''
import torch
import torch.nn as nn
import torch.optim as optim


'''
均方误差损失函数
'''
# 1.初始化
criterion = nn.MSELoss()

# 2.模拟回归值，真实值
predicts = torch.tensor([3.0, 4.0, 5.0])
targerts = torch.tensor([2.8, 4.2, 4.9])

# 3.计算损失
mse_loss = criterion(predicts, targerts)
print(f'MSE Loss: {mse_loss.item():.2f}')


print('\n' + '-' * 50 + '\n')


'''
交叉熵损失函数
'''
# 1.初始化
criterion = nn.CrossEntropyLoss()

# 2.模拟多分类输出 (Logits，未经过Softmax的原始输出)
# shape: (batch_size, num_classes)
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])

# 3.真实标签 (类别索引，如 0, 1, 2)
labels = torch.tensor([0, 1])

# 4.计算损失
cross_entropy_loss = criterion(logits, labels)
print(f'Cross_Entropy Loss: {cross_entropy_loss.item():.2f}')
