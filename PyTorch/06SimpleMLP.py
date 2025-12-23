'''
MLP（简单多层感知机）

至少三层（输入层、一个或多个隐藏层、输出层）的神经元组成，层与层之间全连接

优点：
通用近似器：理论上，一个足够大的MLP可以近似任何连续函数。
结构灵活：可以通过调整层数、神经元数量来适应不同复杂度的问题。
端到端学习：自动从原始数据中学习特征，无需复杂的手工特征工程。

缺点：
全连接导致参数多：输入维度高时（如图像），参数爆炸，计算量大，易过拟合。
不擅长处理序列和空间结构：全连接层会破坏输入数据的拓扑结构（如图像的局部关联、文本的顺序信息）。
对于图像、语音、文本等结构化数据，通常被CNN、RNN、Transformer等更专业的网络取代。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # 定义神经网络
        self.fc1 = nn.Linear(input_size, hidden_size)   # 第一层全连接
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第二层全连接
        self.fc3 = nn.Linear(hidden_size, num_classes)  # 输出层

    def forward(self, x):
        # 前向传播过程
        x = F.relu(self.fc1(x))     # 激活函数用ReLU
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             # 输出层通常不加激活（损失函数会处理）



# 实例化模型
model = SimpleMLP(input_size=784, hidden_size=128, num_classes=10)
print(model)



