import numpy as np


'''
特征矩阵 特征值（只适用于方阵）
'''
A = np.array([
    [1, 2, 1],
    [2, 1, 1],
    [1, 1, 2]
])

# 求解：特征值 特征向量
eigen_value, eigen_vector = np.linalg.eigh(A)

# 特征值
print('\n特征值：')
for i, lambda_i in enumerate(eigen_value):
    print(f'特征值{i + 1} = {lambda_i}（类型：{type(lambda_i)}）')

# 特征向量
print('\n特征向量：')
print(eigen_vector)


# 校验
# A * v = λ * v
for i in range(len(eigen_value)):

    # 对应的特征矩阵（标量）
    lambda_value = eigen_value[i]
    # 第i个特征向量
    v = eigen_vector[:, i]

    # 矩阵乘以特征向量
    left_side = A @ v
    # 特征值（标量）乘以特征向量
    right_side = lambda_value * v

    print(f"特征对 {i + 1}:")
    print(f"A * v = {left_side}")
    print(f"λ * v = {right_side}")
    print(f"是否相等: {np.allclose(left_side, right_side)}")
    print()


print('*' * 50)
print()


'''
反例：
并非方阵的1列对应一个特征矩阵 特征向量
'''
# B = np.array([
#     [1, 1],
#     [0, 1]
# ])
#
# eigen_value, eigen_vector = np.linalg.eigh(B)
#
# for i, lambda_i in enumerate(eigen_value):
#     print(f'特征值{i+1}={lambda_value}（{type(lambda_value)}）')
#
# for i in range(len(eigen_value)):
#     lambda_value = eigen_value[i]
#     v = eigen_vector[:, i]
#
#     left_side = B @ v
#     right_side = lambda_value * v
#
#     print(f'B * v = {left_side}')
#     print(f'λ * v = {right_side}')
#     print('是否相等：%s' % (np.allclose(left_side, right_side)))





