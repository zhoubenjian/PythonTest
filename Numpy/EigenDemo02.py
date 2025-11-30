import numpy as np


A = np.array([
    [1, 1, 2],
    [2, 1, 1],
    [1, 0, 2]
])

eigen_values, eigen_vectors = np.linalg.eig(A)
print(eigen_values.shape)   # (3,)
print(eigen_values.ndim)    # 1


# 验证
for i in range(len(eigen_values)):
    v = eigen_vectors[:, i]
    lambda_i = eigen_values[i]

    left_side = A @ v
    right_side = lambda_i * v

    print(f"特征对 {i + 1}:")
    print(f"A * v = {left_side}")
    print(f"λ * v = {right_side}")
    print(f"是否相等: {np.allclose(left_side, right_side)}")
