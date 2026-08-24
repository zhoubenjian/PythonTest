'''
期望（Expectation）：中心在哪

方差（Variance）：散得有多开

协方差（Covariance）：两个变量如何联动
'''
import numpy as np


# 均值5 标准差2
data = np.random.normal(5, 2, 10000)
print(f'均值={np.mean(data):.3f}，方差={np.var(data):.3f}，标准差={np.std(data):.3f}')


# 协方差   标准正态分布(μ=0, σ²=1)
x = np.random.randn(1000)
# 正相关
y1 = 0.8 * x + np.random.randn(1000) * 0.3
# 负相关
y2 = -0.8 * x + np.random.randn(1000) * 0.3


# 协方差矩阵
data_3d = np.random.randn(100, 3)
print('\n协方差矩阵：\n', np.round(np.cov(data_3d.T), 3), sep='')
# 协方差矩阵是对称阵 转置后等于本身
print('对称：', np.allclose(np.cov(data_3d.T), np.cov(data_3d.T).T), sep='')
