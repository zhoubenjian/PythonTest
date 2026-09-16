'''
L2正则化（L2 Regularization）：在损失函数里，给模型权重的平方加一个惩罚项，不让权重变得太大。

    Loss = 1/n * Σ (y_i - y_pred_i)^2 + λ * Σ w_i^2
        λ≥0：惩罚系数（岭回归里叫α）
        w_i：模型各个权重（系数）想
        Σ w_i^2：权重平方求和，就是 L2 正则项

    模型处理两件事：
    1. 尽量减小预测误差（拟合数据）
    2. 尽量减小所有权重的平方（权重不能太大）
'''
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


# 构造共线数据
X = np.array([
    [1, 1],
    [2, 2],
    [3, 3]
])


# 标准化（L2必须标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = np.array([2, 4, 6])


# OLS普通线性回归
ols = LinearRegression()
ols.fit(X_scaled, y)
print(f'OLS系数(斜率): {ols.coef_}')
print(f'OLS截距: {ols.intercept_:.2f}')

# 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
print(f'岭回归系数(斜率): {ridge.coef_}')
print(f'岭回归截距: {ridge.intercept_:.2f}')
