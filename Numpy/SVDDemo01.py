'''
特征向量 特征值
只适用于方阵

SVD（奇异值分解）
适用于所有矩阵（不局限于方阵）
'''
import numpy as np


# 定义一个 2×2 矩阵
A = np.array([
    [2, 1],
    [1, 2]
])

eigen_value, eigen_vector = np.linalg.eigh(A)

# 特征值
print('\n特征值：')
for i, lambda_i in enumerate(eigen_value):
    print(f'特征值{i + 1} = {lambda_i}（类型：{type(lambda_i)}）')

# 特征向量
print('\n特征向量：')
print(eigen_vector)
print()

# 校核
for i in range(len(eigen_value)):
    lambda_value = eigen_value[i]   # 对应的特征矩阵（标量）
    v = eigen_vector[:, i]

    left_side = A @ v
    right_side = lambda_value * v

    print(f"特征对 {i + 1}:")
    print(f"A * v = {left_side}")
    print(f"λ * v = {right_side}")
    print(f"是否相等: {np.allclose(left_side, right_side)}")
    print()


print('=' * 60)


# 执行 SVD 分解
# U: 左奇异向量矩阵（正交）
# S: 奇异值数组（降序排列）
# Vt: 右奇异向量矩阵的转置（正交）
U, S, Vt = np.linalg.svd(A)
print(f'\n左奇异向量矩阵 U：')
print(U)
print(f'\n奇异值数组：{S}')
print(f'\n右奇异向量转置 Vt：')
print(Vt)
print('-' * 30)


# 验证：U·Σ·Vt 是否等于原始矩阵 A
# 注意：S 是一维数组，需要转成对角矩阵
Sigma = np.diag(S)
A_reconstruct = U @ Sigma @ Vt
print("重构后的矩阵：")
print(A_reconstruct)
print("是否和原矩阵相等？", np.allclose(A, A_reconstruct))  # 浮点精度验证


