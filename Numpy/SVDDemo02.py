import numpy as np


A = np.array([
    [1, 2, 3],
    [2, 1, 1]
])

U, S, Vt = np.linalg.svd(A)
print(f'U:{U}')     # 左奇异向量矩阵
print()
print(f'S:{S}')     # 奇异值数组
print()
print(f'Vt:{Vt}')   # 右奇异向量转置

# 校验
sigma = np.zeros_like(A, dtype=np.float64)  # 创建与A同型的0矩阵
np.fill_diagonal(sigma, S)                  # 奇异值填充
print("完整的 Σ 矩阵（2×3）：")
print(sigma)
print('-' * 60)

A_reconstruct = U @ sigma @ Vt
print('重构后的矩阵：')
print(A_reconstruct)
print(f'重构后是否相等：{np.allclose(A, A_reconstruct)}')

error = np.abs(A - A_reconstruct)
print("\n逐元素误差（绝对值）：")
print(error)