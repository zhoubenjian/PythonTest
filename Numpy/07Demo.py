'''
奇异值分解
'''
import numpy as np


A = np.array([
    [3, 2, 2],
    [2, 3, -2]
], dtype=float)

# 奇异值分解
U, s, VT = np.linalg.svd(A)
print('s =', s)
# 创建与 A 同型的零矩阵
Sigma = np.zeros_like(A, dtype=float)
# 将 s 填入主对角线
np.fill_diagonal(Sigma, s)
print('Sigma = \n', Sigma)

print('*' * 30)

print('U = \n', U)
print('Singular values = ', s)
print('V^T = \n', VT)

print("\n重构的矩阵 A_reconstructed:")
A_reconstructed = U @ Sigma @ VT
print(A_reconstructed)

# 校验误差
print(f'重构误差：{np.linalg.norm(A - A_reconstructed):.10f}')