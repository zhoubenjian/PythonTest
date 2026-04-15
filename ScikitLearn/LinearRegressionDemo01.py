'''
线性回归（预测连续值）
做回归任务，预测连续值

任务类型：回归（预测连续值）
输出结果：任意实数（-∞ ~ +∞）
核心公式：y=wx+b
激活函数：无
损失函数：均方误差 MSE
假设分布：高斯分布
典型场景：房价、销量、温度预测
'''
import numpy as np
from sklearn.linear_model import LinearRegression


# 1.模拟数据
# X: 输入特征
X = np.array([
    [1], [2], [3], [4], [5]
])

# y: 目标变量
y = np.array([
    2, 4, 6, 8, 10
])


# 2.实例化模型
model = LinearRegression()
# 训练模型
model.fit(X, y)


# 3.预测结果
y_pred = model.predict([[6]])
print('预测值：{:.1f}'.format(y_pred[0]))       # 12.0

