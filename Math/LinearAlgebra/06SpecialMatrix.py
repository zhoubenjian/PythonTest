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
print('b @ [1, 1, 1] = ', b @ c, sep='')    # b @ [1, 1, 1] = [2 3 5]


print('\n' + '=' * 30 + '\n')


# 对称阵（Symmetric matrix）
d = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])
print('d是否对称：', np.allclose(d, d.T), sep='')                  # d是否对称：True

# 这是最重要的性质！！！对称矩阵的所有特征值都是实数（不是复数）
eigenvals = np.linalg.eigvals(d)
# print(eigenvals)
print('对称矩阵特征值全是实数：', np.all(np.isreal(eigenvals)), sep='')    # True

