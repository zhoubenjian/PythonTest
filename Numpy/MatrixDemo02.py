'''
伴随矩阵和逆矩阵
'''
import numpy as np


"""
    计算方阵的伴随矩阵（numpy实现）
    :param matrix: 输入方阵（numpy数组）
    :return: 伴随矩阵（numpy数组）
"""
# 求伴随矩阵（矩阵的代数余子式转置）
def adjugate_matrix(matrix):

    # 1.增强鲁棒性
    matrix = np.array(matrix, dtype=np.float64)

    # 2.校验输入是否为方阵
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("输入必须是二维方阵！")

    # 3.初始化代数余子式矩阵
    n = matrix.shape[0]
    cofactor_mat = np.zeros_like(matrix, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            # 去掉第i行，第j列，得到余子式
            sub_mat = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            # 计算代数余子式：(-1)^(i+j) * 子矩阵行列式
            cofactor = ((-1) ** (i + j)) * np.linalg.det(sub_mat)
            cofactor_mat[i, j] = cofactor

    # 4.伴随矩阵 = 代数余子式矩阵的转置
    adj_mat = cofactor_mat.T
    return adj_mat


# 验证
# 1.定义一个非奇异方阵（行列式≠0）
A = np.array([
    [2, 3, 5],
    [7, 11, 13],
    [17, 19, 23]
], dtype=np.float64)


# 2. 计算行列式（验证可逆性）
det_A = np.linalg.det(A)
print(f'矩阵A的行列式：{det_A:.2f}')       # 矩阵A的行列式：-78.00


# 3.计算伴随矩阵
adj_A = adjugate_matrix(A)
print('\n矩阵A的伴随矩阵：')
# 四舍五入避免浮点误差
print(np.round(adj_A, 2))


'''
4.计算逆矩阵
'''
# 方法1：通过伴随矩阵计算逆矩阵（逆矩阵 = 伴随矩阵 / 行列式）
inv_A_by_adj = adj_A / det_A
print('\n通过伴随矩阵计算逆矩阵：')
print(np.round(inv_A_by_adj, 2))

print('*' * 30)

# 方法2：Numpy内置函数直接求逆矩阵（工程首选）
inv_A_direct = np.linalg.inv(A)
print("numpy内置函数计算的逆矩阵：")
print(np.round(inv_A_direct, 2))


# 6.验证：原矩阵 × 逆矩阵 ≈ 单位矩阵
verify = np.dot(A, inv_A_by_adj)
print('\n验证逆矩阵（伴随矩阵求逆矩阵）（A × A⁻¹）：')
print(np.round(verify, 2))

print('*' * 40)

verify = np.dot(A, inv_A_direct)
print('验证逆矩阵（numpy内置函数求逆矩阵）（A × A⁻¹）：')
print(np.round(verify, 2))







