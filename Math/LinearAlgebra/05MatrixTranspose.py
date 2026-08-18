'''
矩阵转置
    行列互换：原来 (i, j) 位置的值，经过转置后到了 (j, i) 位置。 形状从 m×n 变成 n×m
'''
import numpy as np


'''
转置（行列互换）：原来i, j) 位置的值，经过转置后到了 (j, i) 位置。 形状从 m×n 变成 n×m
'''
a = np.array([[1, 2, 3], [4, 5, 6]])
print(f'a(2x3):\n{a}')
print(f'\na.T(3x2):\n{a.T}')


print('\n' + '*' * 30 + '\n')


# 验证(AB)^T = B^T @ A^T
b = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
ab = a @ b
''' 
[[22 49]
 [28 64]]
'''
print(f'ab.T:\n{ab.T}')
print(f'是否相等：{np.allclose(ab.T, b.T @ a.T)}')   # True


print('\n' + '*' * 30 + '\n')


# 逆矩阵（只有方阵才有逆矩阵！！！）
c = np.array([
    [2, 1],
    [5, 3]
])
c_inv = np.linalg.inv(c)
print(f'c:\n{c}')
print(f'\nc_inv:\n{c_inv}')
print(f'\nc @ c_inv:\n{c @ c_inv}')

print(f'\nc @ c_inv是否相等单位阵(主对角线为1，其他为0)：{np.allclose(c @ c_inv, np.eye(2))}')   # True


'''
奇异矩阵（Singular matrix）没有逆矩阵
'''
# det(d)=0, d没有逆矩阵
d = np.array([
    [1, 2],
    [2, 4]
])

try:
    d_inv = np.linalg.inv(d)
except np.linalg.LinAlgError:
    print('det(d)=0, d没有逆矩阵')

