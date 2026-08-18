'''
特殊矩阵
    单位阵（Identity matrix）
    对角阵（Diagonal matrix）
    对称阵（Symmetric matrix）
'''
import numpy as np


# 单位矩阵（Identity matrix）
I = np.eye(3)
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f'I @ a == a：{np.allclose(I @ a, a)}')    # True
print(f'a @ I == a：{np.allclose(a @ I, a)}')    # True


print('\n' + '=' * 30 + '\n')


'''
对角阵（Diagonal matrix）
    独立放缩
'''
b = np.diag([2, 3, 5])
c = np.array([1, 1, 1])
print('b @ [1, 1, 1] = ', b @ c, sep='')     # b @ [1, 1, 1] = [2 3 5]


print('\n' + '=' * 30 + '\n')


# 对称阵（Symmetric matrix）
d = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])
print('d是否对称：', np.allclose(d, d.T), sep='')                  # d是否对称：True


'''
对称矩阵的所有特征值都是实数（不是复数）
！！！这是最重要的性质！！！
'''
# 求取d的所有特征值
eigenvals = np.linalg.eigvals(d)
print('对称矩阵特征值全是实数：', np.all(np.isreal(eigenvals)), sep='')    # True


'''
协方差矩阵（Covariance matrix）天然对称
'''
# 生成一个形状为 (100, 5) 的随机矩阵
# 100个样本（行），每个样本有5个特征（列） 数据服从标准正态分布（均值为0，方差为1）
data = np.random.randn(100, 5)

# 默认把行当作特征，列当作样本，所以必须转置；或使用rowvar=False
# cov_matrix = np.cov(data.T)
cov_matrix = np.cov(data, rowvar=False)

print('协方差矩阵对称：', np.allclose(cov_matrix, cov_matrix.T), sep='')    # True


