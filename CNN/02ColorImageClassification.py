'''
使用 CNN 对 CIFAR-10 的 10 类彩色小图像（32×32）进行分类。
'''


import torch
import torch.nn as nn                           # 神经网络模块
import torch.optim as optim                     # 优化器
import torchvision                              # 计算机视觉数据集和模型
import torchvision.transforms as transforms     # 图像预处理
from torch.utils.data import DataLoader         # 数据加载器

# 2.数据预处理与加载
# 训练集
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # 随机裁剪
    transforms.RandomHorizontalFlip(),           # 随机水平翻转
    transforms.ToTensor(),                       # 转为张量
    transforms.Normalize(                        # 标准化
        (0.4914, 0.4822, 0.4465),          # RGB通道的均值
          (0.2023, 0.1994, 0.2010)           # RGB通道的标准差
    ),
])

# 测试集（注意：测试集不进行数据增强，只做必要转换）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

# 加载训练数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data',              # 数据存储路径
    train=True,                 # 加载训练集（5万张）
    download=True,              # 如果不存在则下载
    transform=transform_train   # 应用预处理
)
trainloader = DataLoader(
    trainset,                   # 数据集对象
    batch_size=128,             # 每批128张图像
    shuffle=True,               # 每个epoch随机打乱
    num_workers=2)              # 并行加载数据的进程数

# 加载测试数据集
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,                # 加载测试集（1万张）
    download=True,
    transform=transform_test
)
testloader = DataLoader(
    testset,
    batch_size=100,             # 测试时批大小为100
    shuffle=False,              # 测试时不打乱
    num_workers=2
)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# 3.定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature = nn.Sequential(
            # --- 第一卷积块 ---
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # 输入:3通道, 输出:32通道
            nn.ReLU(),                                                            # 激活函数
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 32 => 64通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                # 下采样:32×32 => 16×16

            # --- 第二卷积块 ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # 64 => 128通道
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                # 下采样:16×16 => 8×8

            # --- 第三卷积块 ---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 128 => 256通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                                 # 下采样:8×8 => 4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    # 前向传播
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x


# 4.设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5.训练与验证函数
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(transform_train), 100 * correct / total

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


# 6.开始训练（例如 10 个 epoch）
train_losses = []
train_accs = []
test_accs = []

num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch()
    test_acc = evaluate()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Loss: {train_loss:.4f} | '
          f'Train Acc: {train_acc:.2f}% | '
          f'Test Acc: {test_acc:.2f}%')


