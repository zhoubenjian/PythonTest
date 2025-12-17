'''
PyTorch实现简单卷积神经网络（CNN）
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一个卷积层，输入通道数为1（灰度图像），输出通道数为32，卷积核大小为3 × 3，步长为1，填充为1。
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # 第二个卷积层，输入通道数为32（灰度图像），输出通道数为64，卷积核大小为3 × 3，步长为1，填充为1。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 第一个全连接层，输入大小为NBSP64 × 7 × 7，输出大小为128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # 第二个全连接层，输入大小为128，输出大小为10（对应10个类别）
        self.fc2 = nn.Linear(128, 10)

    # 定义数据的前向传播过程
    def forward(self, x):
        x = F.relu(self.conv1(x))           # 通过 conv1 进行卷积操作，然后使用ReLU激活函数
        x = F.max_pool2d(x, 2)    # 通过最大池化层（max_pool2d）进行下采样，池化窗口大小为 2×22×2
        x = F.relu(self.conv2(x))           # 通过 conv2 进行卷积操作，然后使用ReLU激活函数
        x = F.max_pool2d(x, 2)    # 再次通过最大池化层进行下采样
        x = x.view(-1, 64 * 7 * 7)          # 将特征图展平为一维向量（view），输入到全连接层
        x = F.relu(self.fc1(x))             # 通过 fc1 进行全连接操作，然后使用ReLU激活函数
        x = self.fc2(x)                     # 通过 fc2 输出最终的分类结果
        return x


# 加载数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)   # 加载MNIST训练数据集，设置存储路径为 ./data，如果数据不存在则自动下载
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)          # 创建数据加载器，设置批量大小为64，打乱数据顺序
print(len(train_loader))    # 938

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()                       # 定义损失函数为交叉熵损失（CrossEntropyLoss），适用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=0.001)    # 使用Adam优化器，学习率为0.001

# 训练模型
for epoch in range(5):                                  # 训练5个epoch（即遍历整个数据集5次）
    for batch_idx, (data, target) in enumerate(train_loader):       # 遍历数据加载器，每次获取一个批次的数据（data）和标签（target）
        optimizer.zero_grad()                           # 清零梯度：optimizer.zero_grad()，防止梯度累积
        output = model(data)                            # 前向传播：output = model(data)，计算模型的输出
        loss = criterion(output, target)                # 计算损失：loss = criterion(output, target)，计算模型输出与真实标签的损失
        loss.backward()                                 # 反向传播：loss.backward()，计算梯度
        optimizer.step()                                # 更新参数：optimizer.step()，根据梯度更新模型参数
        if batch_idx % 100 == 0:                        # 打印损失：每100个批次打印一次损失值
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')