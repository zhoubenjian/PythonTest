'''
手写数字识别

实现逻辑：
初始化参数 → 加载数据 → 构建模型 → 设置优化器 → 循环训练(前向传播→计算损失→反向传播→更新权重) → 测试评估 → 输出准确率

MNIST数据集特点：
60,000个训练样本，10,000个测试样本；
28×28像素的灰度手写数字图像；
10个类别（0-9）；
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ----------------------------
# 1. 超参数设置
# ----------------------------
input_size = 28 * 28    # MNIST 图像是 28x28 像素
hidden_size = 128       # 隐藏层神经元数量
num_classes = 10        # 数字 0-9
num_epochs = 5          # 训练5个完整周期
batch_size = 64         # 每批64个样本
learning_rate = 0.001   # Adam优化器的学习率


# ----------------------------
# 2. 数据加载与预处理
# ----------------------------
transform = transforms.Compose([
    # (C, H, W)（通道数、高度、宽度，PyTorch 标准格式）
    transforms.ToTensor(),                      # 转为张量(1,28,28)，[0,1]范围
    transforms.Lambda(lambda x: x.reshape(-1))  # 展平为（784,） 一维张量
])

# 训练集
train_dataset = datasets.MNIST(root='../CNN/data', train=True, transform=transform, download=True)
# 测试集
test_dataset = datasets.MNIST(root='../CNN/data', train=False, transform=transform)

# 加载训练数据
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
print(f'训练数据个数:{len(train_loader)}')    # 训练数据个数:938
# 加载测试数据
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print(f'测试数据个数:{len(test_loader)}')     # 测试数据个数:157


# ----------------------------
# 3. 定义前馈神经网络模型
# ----------------------------
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # 输入层 -> 隐藏层
        self.relu = nn.ReLU()                           # 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 隐藏层 -> 输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # 注意：CrossEntropyLoss内部已包含softmax，输出层不需要额外激活
        return out


# ----------------------------
# 4. 初始化模型、损失函数和优化器
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardNeuralNetwork(input_size, hidden_size, num_classes).to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ----------------------------
# 5. 训练模型
# ----------------------------
print('开始训练')
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 向前传播(自动调用 forward 方法：在 PyTorch 的 nn.Module 基类中，定义了 __call__ 方法)
        output = model(images)              # 获取预测
        loss = criterion(output, labels)    # 计算损失

        # 反向传播与优化
        optimizer.zero_grad()               # 清空梯度
        loss.backward()                     # 反向传播计算梯度
        optimizer.step()                    # 更新参数

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')


# ----------------------------
# 6. 测试模型准确率
# ----------------------------
model.eval()            # 切换到评估模式
with torch.no_grad():   # 不计算梯度，节省内存
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)   # 获取预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'测试准确率: {100 * correct / total:.2f}%')


