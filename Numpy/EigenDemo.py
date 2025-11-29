'''
方阵的特征矩阵，特征向量

对于矩阵 A 和特征向量 v，特征值 λ 满足：
A · v = λ · v

其中：
λ 是标量（一个数）
v 是向量
A 是矩阵
'''
import numpy as np


A = np.array([
    [3, 1],
    [1, 2]
])

# 计算特征值，特征向量
eigen_value, eigen_vector = np.linalg.eigh(A)
# 形状
print(eigen_value.shape)    # (2,)
# 维度
print(eigen_value.ndim)     # 1

print('=' * 50)

print('特征值：')
for i, lambda_i in enumerate(eigen_value):
    print(f'λ{i + 1} = {lambda_i} (类型：{type(lambda_i)})')

print('特征向量：')
print(eigen_vector)


# A * v = λ * v
print('\n验证特指方程：')
for i in range(len(eigen_value)):
    v = eigen_vector[:, i]      # 第i个特征向量
    lambda_i = eigen_value[i]   # 对应的特征矩阵（标量）

    left_side = A @ v           # 矩阵乘以特征向量
    right_side = lambda_i * v   # 特征值（标量）乘以特征向量

    print(f"特征对 {i + 1}:")
    print(f"A * v = {left_side}")
    print(f"λ * v = {right_side}")
    print(f"是否相等: {np.allclose(left_side, right_side)}")

    print()


