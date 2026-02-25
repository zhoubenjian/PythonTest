import numpy as np

'''
特征值，特征向量（只适用于方阵）
'''


def eigen_value_vector(A):
    eigen_value, eigen_vector = np.linalg.eigh(A)

    # 特征值
    print('特征值：')
    for i, lambda_i in enumerate(eigen_value):
        print(f'特征值{i + 1}：{lambda_i}（类型：{type(lambda_i)}）')

    # 特征向量
    print('\n特征向量：')
    print(eigen_value)

    # 校验 A * v = λ * v
    print('\n校验：')
    for i in range(len(eigen_value)):
        # 矩阵乘以特征向量：A * v
        lambda_value = eigen_value[i]
        # 特征值（标量）乘以特征向量：λ * v
        lambda_vector = eigen_vector[:, i]

        left_side = A @ lambda_vector
        right_side = lambda_value * lambda_vector

        print(f'特征对{i + 1}')
        print('A * v = %s' % left_side)
        print(f'λ * v = {right_side}')
        print(f'是否相等：{np.allclose(left_side, right_side)}')

        print('*' * 30)


'''
奇异值分解（适用于所有矩阵）
'''


def svd(B):
    U, S, Vt = np.linalg.svd(B)

    # U: 左奇异向量矩阵（正交）
    print(f'\n左奇异向量矩阵 U：\n{U}')

    # S: 奇异值数组（降序排列）
    print(f'\n奇异值数组：\n{S}')

    # Vt: 右奇异向量矩阵的转置（正交）
    print(f'\n右奇异向量转置 Vt：\n{Vt}')

    # 校验: U·Σ·Vt 是否等于原始矩阵 B
    sigma = np.zeros_like(B, dtype=np.float64)  # 创建与B同型的零矩阵
    np.fill_diagonal(sigma, S)  # 奇异值填充
    print('\n完整的 Σ 矩阵（2×3）：')
    print(sigma)

    B_reconstruct = U @ sigma @ Vt
    print('\n重构B矩阵：')
    print(B_reconstruct)
    print(f'重构后是否相等：{np.allclose(B, B_reconstruct)}')

    error = np.abs(B, B_reconstruct)
    print("\n逐元素误差（绝对值）：")
    print(error)


if __name__ == "__main__":
    '''
    特征向量，特征值
    '''
    A = np.array([
        [1, 2],
        [2, 1]
    ])

    eigen_value_vector(A)

    print('\n' + '#' * 50)

    '''
    奇异值分解
    '''
    B = np.array([
        [1, 2, 1],
        [2, 1, 1]
    ])
    svd(B)
