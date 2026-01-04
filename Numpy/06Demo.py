import numpy as np
import matplotlib.pyplot as plt


# 创建一个示例矩阵
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

print(f'原始矩阵A（4 x 3）：')
print(A)
print(f'A矩阵形状：{A.shape}')
print(f'A矩阵的秩：{np.linalg.matrix_rank(A)}')

# 进行奇异值分解
U, S, VT = np.linalg.svd(A, full_matrices=True)

print("\n=== SVD 分解结果 ===")
print(f"U 矩阵形状: {U.shape}")
print(f"S (奇异值向量) 形状: {S.shape}")
print(f"VT 矩阵形状: {VT.shape}")

print("\n奇异值 S:")
print(S)

print("\n前 3 个左奇异向量 (U 的前 3 列):")
print(U[:, :3])

print("\n右奇异向量矩阵 VT:")
print(VT)

# 验证分解的正确性：重构原矩阵
# 注意：需要将奇异值向量转换为对角矩阵
Sigma = np.zeros(A.shape)
for i in range(len(S)):
    Sigma[i, i] = S[i]

A_reconstructed = U @ Sigma @ VT
print("\n重构的矩阵 A_reconstructed:")
print(A_reconstructed)
print(f"\n重构误差 (Frobenius 范数): {np.linalg.norm(A - A_reconstructed):.10f}")
