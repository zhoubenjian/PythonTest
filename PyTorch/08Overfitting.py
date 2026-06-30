'''
一言以蔽之：
过拟合：模型学得太细、太死，泛化能力差
正则化：通过约束模型复杂度，让它学得更通用，提升泛化能力

过拟合演示
L2正则化对比

过拟合(训练好、测试很差)：方差大，偏差小
欠拟合(训练差、测试差)：方差小，偏差大
正则化(训练适中、测试优秀)：增加一点偏差，大幅降低方差

常见正则化方法:
- 1.L1正则化：对权重的绝对值进行惩罚，鼓励稀疏解
    特点：容易让部分参数直接变为 0
    作用：特征选择，稀疏化模型

- 2.L2正则化（Ridge / 权重衰减）
    特点：让所有参数都变小，但不会轻易变 0
    作用：让模型更平滑，最常用

- 3.Dropout 正则化（随机丢弃神经元）
    训练时随机 “关掉” 一部分神经元，强迫模型不依赖某些特征，防止过拟合。

- 4.早停（Early Stopping）
    验证集误差不再下降时就停止训练，避免继续拟合噪声。
'''
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# 强制切换后端，解决PyCharm绘图报错
matplotlib.use('TkAgg')  # 关键修复
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

# 2.带L2正则化(weight_decay)    正则化 = 牺牲一点训练精度，换更好的泛化能力
net_l2 = OverfitNet()
opt_l2 = optim.Adam(net_l2.parameters(), lr=0.005, weight_decay=0.01)    # weight_decay把权重往0拉，让模型更平滑

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
    pred_l2 = net_l2(x)
    loss_l2 = criterion(pred_l2, y)
    opt_l2.zero_grad()                  # 清空旧梯度
    loss_l2.backward()                  # 计算新梯度
    opt_l2.step()                       # 更新参数


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