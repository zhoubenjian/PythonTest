'''
前向传播（Forward Propagation），反向传播（Backward Propagation）
'''
import torch
import torch.nn as nn
import torch.optim as optim


# 1.定义一个简单的全连接神经网络（输入层 => 隐藏层 => 输出层）
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 定义可学习参数（权重和偏置，PyTorch自动管理）
        self.fc1 = nn.Linear(input_size, hidden_size)   # 第一层：线性变换 W1x + b1
        self.relu = nn.ReLU()                           # 激活函数（非线性变换）
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第二层：线性变换 W2x + b2

    # 2. 定义前向传播流程（必须实现forward方法）
    def forward(self, x):
        # 前向传播：输入 => 第一层 => 激活 => 第二层 => 输出（预测值）
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 3. 初始化网络、损失函数、优化器
input_size = 6      # 输入特征维度（对应你之前6x6图像的一行特征）
hidden_size = 12    # 隐藏层（中间层）
out_size = 1        # 输出维度（回归任务，预测一个标量）

# 实例化全连接神经网络
net = SimpleNet(input_size, hidden_size, out_size)
# 损失函数：均方误差（适用于回归任务）
criterion = nn.MSELoss()
# SGD优化器：随机梯度下降，学习率0.01
optimizer = optim.SGD(net.parameters(), lr=0.01)


# 4. 构造模拟数据（输入x和真实标签y）
x = torch.randn(10, input_size, dtype=torch.float32)    # 10个样本，每个样本6个特征
y_true = torch.randn(10, out_size, dtype=torch.float32) # 10个样本的真实标签


# 5. 单轮训练（前向传播 => 反向传播 => 参数更新）
print('***** 训练前，网络第一层权重（部分） *****')
# 查看训练前的权重（用于对比更新效果）
print(net.fc1.weight)
print(net.fc1.weight[:2, :2])

# （1）前向传播：计算预测值和损失
y_pred = net(x)                     # 传入输入数据，自动执行forward方法，得到预测值
loss = criterion(y_pred, y_true)    # 计算预测值与真实值的损失
print(f"\n===== 前向传播完成，损失值：{loss.item():.4f} =====")

# （2）反向传播：计算梯度（注意：反向传播前需清空上一轮梯度）
optimizer.zero_grad()   # 清空梯度缓存（PyTorch梯度会累加，必须手动清空）
loss.backward()         # 自动执行反向传播，计算所有可学习参数的梯度（核心方法）
print("\n===== 反向传播完成，网络第一层权重梯度（部分） =====")
print(net.fc1.weight.grad[:2, :2])  # 查看反向传播得到的梯度

# （3）参数更新：优化器利用梯度更新参数
optimizer.step()        # 自动执行参数更新，基于反向传播得到的梯度
print("\n===== 参数更新完成，网络第一层权重（部分） =====")
print(net.fc1.weight[:2, :2])       # 查看更新后的权重（与训练前对比，已发生变化


