'''
加利福尼亚房价回归预测（连续值输出）
    8项房屋 / 周边特征，预测房价（回归任务）
    加利福尼亚房价回归预测（连续值输出）
'''


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


'''
1.加载数据集
'''
housing = fetch_california_housing()
# print(housing.feature_names)        # 收入中位数:'MedInc', 房屋年龄中位数:'HouseAge', 平均房间数:'AveRooms', 平均卧室数:'AveBedrms', 人口数量:'Population', 户均人口数:'AveOccup', 纬度:'Latitude', 经度:'Longitude'
# print(housing.target_names)         # MedHouseVal(房价中位数)
X = housing.data
y = housing.target.reshape(-1, 1)

# 标准化
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


'''
2.回归MLP
'''
class CaliforniaHouseMLP(nn.Module):
    # 初始化模型
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    # 前向传播
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# 实例化模型
model = CaliforniaHouseMLP()
# 定义均方误差函数(回归任务)
criterion = nn.MSELoss()
# 定义Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)


'''
3.训练模型
'''
epochs = 200
for epoch in range(epochs):
    # 切换到训练模式(对于回归任务，训练模式与评估模式没有区别)
    model.train()
    # 预测训练集结果
    pred = model(X_train)
    # 计算训练集损失
    train_loss = criterion(pred, y_train)     # criterion(预测值, 真实值)

    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    train_loss.backward()
    # 更新参数权重
    optimizer.step()

    # 测试集打印损失(每20个epoch打印一次)
    if (epoch + 1) % 20 == 0:
        # 切换到评估模式(对于回归任务，评估模式与训练模式没有区别)
        model.eval()
        # 关闭梯度计算，大幅节省显存、提速
        with torch.no_grad():
            # 预测测试集结果
            test_pred = model(X_test)
            # 计算测试集损失
            test_loss = criterion(test_pred, y_test)
        print(f'Epoch {epoch+1:3d} | Train MSE Loss: {train_loss.item():.4f} | Test MSE Loss: {test_loss.item():.4f}')


