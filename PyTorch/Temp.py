'''
信用卡逾期预测
'''
import torch
import torch.nn as nn
import torch.optim as optim


# 1.定义CreditFNN模型
class CreditFNN(nn.Module):

    # 初始化
    def __init__(self):
        super().__init__()

        # 全连接层1：输入层（5）=> 输出层（8）
        self.fc1 = nn.Linear(5, 8)
        # 全连接层2：输入层（8）=> 输出层（4）
        self.fc2 = nn.Linear(8, 4)
        # 全连接层2：输入层（4）=> 输出层（1）
        self.fc3 = nn.Linear(4, 1)

        self.relu = nn.ReLU()           # Relu常用于隐藏层

        self.sigmoid = nn.Sigmoid()     # 通常用于二分类的输出层


    # 前向传播
    def forward(self, x):
        # 第一层：线性变换 => ReLU激活
        x = self.relu(self.fc1(x))

        # 第二层：线性变换 => ReLU激活
        x = self.relu(self.fc2(x))

        '''
        第三层（输出层）：线性变换 => Sigmoid激活
        Sigmoid将输出压缩到(0, 1)区间，表示离职概率
        '''
        x = self.sigmoid(self.fc3(x))

        # 返回预测结果，形状 [batch_size, 1]
        return x


# 2.准备训练数据
# 特征矩阵: 6个样本，每个5个特征
X = torch.tensor([
    [25.5, 0.35, 2, 680, 32],
    [12.0, 0.60, 4, 550, 25],
    [50.0, 0.20, 0, 780, 45],
    [18.0, 0.55, 3, 590, 28],
    [8.0, 0.70,  5, 520, 22],
    [30.0, 0.30, 1, 720, 35]
], dtype=torch.float32)

# 标签: 1=逾期, 0=按时
y = torch.tensor([
    [0],
    [1],
    [0],
    [1],
    [1],
    [0]
], dtype=torch.float32)


# 3.初始化模型、损失函数、优化器
model = CreditFNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 4.训练循环
epochs = 1000
for epoch in range(epochs):

    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100轮打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch[{epoch + 1} / {epochs}], Loss:{loss.item():.4f}')


# 5.预测新客户
new_customer = torch.tensor([
    [15.0, 0.50, 2, 620, 26]
], dtype=torch.float32)

with torch.no_grad():
    prob = model(new_customer)
    print(f'新客户信用卡预期概率：{prob.item():.3f}')
    print(f"预测结果：{'逾期风险高' if prob.item() > 0.5 else '信用良好'}")







