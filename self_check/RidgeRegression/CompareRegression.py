'''
 L2岭回归（Ridge Regression） vs 普通线性回归（几何对比）
    岭回归 = 线性回归 + L2 正则化。它通过"惩罚大的权重"，防止模型过拟合。
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

import common.mpl_config


'''
1. 生成模拟数据（带多重共线）
'''
np.random.seed(42)
# 50个样本，10个特征
n_samples = 50
X = np.random.randn(n_samples, 10)
# 制造多重共线性：第2个特征是第1个的近似复制
X[:, 1] = X[:, 0] + 0.01 * np.random.randn(n_samples)
# 真实关系
true_w = np.array([1.5, -2.0, 0.5, 0, 0, 0, 0, 0, 0, 0])
y = X @ true_w + 0.5 * np.random.randn(n_samples)


'''
2. 普通线性回归
'''
lr = LinearRegression()
lr.fit(X, y)


'''
3. 岭回归
'''
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)


'''
4. 对比权重
'''
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(lr.coef_))
width = 0.35

plt.bar(x_pos - width/2, lr.coef_, width, label='普通线性回归', color='red', alpha=0.7)
plt.bar(x_pos + width/2, ridge.coef_, width, label='岭回归 (α=1.0)', color='blue', alpha=0.7)
plt.axhline(y=0, color='black', linewidth=1)
plt.xlabel('特征索引')
plt.ylabel('权重值')
plt.title('普通线性回归 vs 岭回归：权重对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
