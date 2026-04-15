'''
波士顿房价预测
'''
import torch
import torch.nn as nn
import torch.optim as optim


# 1.定义FNN模型(回归问题)
class BostonHousingFNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 全连接层1：输入层(5) => 隐藏层1(8)
        self.fc1 = nn.Linear(5, 8)

        # 全连接层2：隐藏层1(8) => 隐藏层2(4)
        self.fc2 = nn.Linear(8, 4)

        # 全连接层3：隐藏层2(4) => 输出层(1)
        self.fc3 = nn.Linear(4, 1)

        # 注意：回归问题输出层不用Sigmoid，因为房价可以是任意正数!!!
        # ReLU输出范围: max(0, x)
        self.relu = nn.ReLU()

    # 前向传播
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 2.准备数据
# 特征矩阵: 6个样本，每个5个特征
X = torch.tensor([
    [6.2, 65.0, 320, 12.5, 15.3],   # 样本1
    [5.0, 85.0, 400, 25.0, 18.5],   # 样本2
    [7.5, 20.0, 250, 5.0, 14.0],    # 样本3
    [4.5, 95.0, 500, 30.0, 20.0],   # 样本4
    [6.8, 45.0, 280, 10.0, 16.0],   # 样本5
    [5.5, 70.0, 350, 18.0, 17.5]    # 样本6
], dtype=torch.float32)

# 标签: 房价（千美元）- 注意是连续值，不是0/1
y = torch.tensor([
    [24.5],     # 样本1: 房价24.5
    [15.0],     # 样本2: 房价15.0
    [35.0],     # 样本3: 房价35.0
    [12.0],     # 样本4: 房价12.0
    [28.0],     # 样本5: 房价28.0
    [18.0]      # 样本6: 房价18.0
], dtype=torch.float32)


# 3.初始化模型，损失函数，优化器
model = BostonHousingFNN()
# 关键区别1: 使用 MSELoss（均方误差）而不是 BCELoss
# 公式: Loss = (y_pred - y_true)² 的平均值
criterion = nn.MSELoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 4.训练循环
epochs = 1000
for epoch in range(epochs):

    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播
    optimizer.zero_grad()   # 清空旧梯度
    loss.backward()         # 计算新梯度
    optimizer.step()        # 更新参数

    # 每100轮打印损失
    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# 5.预测新房屋价格
new_house = torch.tensor([
    [6.0, 50.0, 300, 15.0, 16.0]
], dtype=torch.float32)

with torch.no_grad():       # 测试时不需要计算梯度
    price = model(new_house)
    print(f'预测房价：${price.item():.2f}K')
    print(f'即$:{price.item() * 1000:.2f}')



