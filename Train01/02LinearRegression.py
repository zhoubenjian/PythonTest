'''
回归问题
'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([3, 5, 7, 9, 11])

# 创建模型
lr = LinearRegression()
# 训练模型
lr.fit(X_train, y_train)

# Coefficient（系数/斜率)）
print(f'斜率：{lr.coef_[0]:.2f}')
# Intercept (截距)
print(f'截距：{lr.intercept_:.2f}')

# 预测
x_test = np.array([[10], [20]])     # 测试特征值
y_true = np.array([21, 42])         # 测试标签（真实值）
y_pre = lr.predict(x_test)          # 预测值

# 模型评估
print(f'均方误差：{mean_squared_error(y_true, y_pre):.2f}')     # 均方误差越小拟合越好
# R² = 1 - (MSE/方差）
print(f'R²分数：{r2_score(y_true, y_pre):.2f}')                # R²越接近于1，拟合效果越好



# 可视化
plt.figure(figsize=(12, 8))
plt.plot(X_train, y_train, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'y = {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}')
plt.show()

