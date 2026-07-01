'''
鸢尾花三分类（极简多分类）
'''


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1.加载数据集
iris = load_iris()
# print(iris.feature_names)   # 花萼长度:'sepal length (cm)', 花萼宽度:'sepal width (cm)', 花瓣长度:'petal length (cm)', 花瓣宽度:'petal width (cm)'
# print(iris.target_names)    # 山鸢尾0（setosa）、变色鸢尾1（versicolor）、维吉尼亚鸢尾2（virginica）
X = iris.data
y = iris.target

# 2.归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.转为张量类型
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# 4.定义模型
class IrisMLP(nn.Module):
    # 初始化模型
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    # 前向传播
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.relu(self.fc2(x))

# 实例化模型
model = IrisMLP()
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义Adam优化器
optim = optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    # 前向传播
    out = model(X_train)
    # 计算损失
    loss = criterion(out, y_train)
    # 梯度清零
    optim.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数权重
    optim.step()
    if (epoch + 1) % 20 == 0:
        pred = torch.argmax(model(X_test), dim=1)
        acc = (pred == y_test).sum() / len(y_test)
        print(f"Epoch {epoch+1}, Loss:{loss:.3f}, Acc:{acc:.3f}")





