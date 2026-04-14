'''
过拟合演示
L2正则化对比

过拟合：方差大，偏差小
欠拟合：方差小，偏差大
正则化：增加一点偏差，大幅降低方差
'''
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 生成带噪声的数据
x = torch.unsqueeze(torch.linspace(-2, 2, 300), dim=1)
y = x ** 2 + 0.5 * torch.randn(x.size())


# 超复杂网络(极易过拟合)
class OverfitNet(nn.Module):
    # 初始化
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    # 前向传播
    def forward(self, x):
        return self.fc(x)


# 1.无正则化(过拟合版本)
net_no_reg = OverfitNet()
net_no_reg_optimizer = optim.Adam(net_no_reg.parameters(), lr=0.005)

# 2.带L2正则化(weight_decay)
net_12 = OverfitNet()
opt12 = optim.Adam(net_12.parameters(), lr=0.005, weight_decay=0.01)    # weight_decay把权重往0拉，让模型更平滑

# 均方误差损失函数
criterion = nn.MSELoss()


# 训练
epochs = 2000
for _ in range(epochs):
    # 无正则化
    pred_no = net_no_reg(x)
    loss_no = criterion(pred_no, y)
    net_no_reg_optimizer.zero_grad()    # 清空旧梯度
    loss_no.backward()                  # 计算新梯度
    net_no_reg_optimizer.step()         # 更新参数

    # L2正则
    pred_l2 = net_12(x)
    loss_l2 = criterion(pred_l2, y)
    opt12.zero_grad()                   # 清空旧梯度
    loss_l2.backward()                  # 计算新梯度
    opt12.step()                        # 更新参数


# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y, s=10, alpha=0.5, label='data')
plt.plot(x, pred_no.detach(), 'r-', linewidth=2, label='no reg (overfit)')
plt.title('Overfitting (No Regularization)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x, y, s=10, alpha=0.5, label='data')
plt.plot(x, pred_l2.detach(), 'g-', linewidth=2, label='L2 reg')
plt.title('L2 Regularization (Smoother)')
plt.legend()

plt.tight_layout()
plt.show()