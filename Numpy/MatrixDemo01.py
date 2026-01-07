import numpy as np


"""
    计算矩阵的行列式（包含输入校验）
    :param matrix: 输入矩阵（可以是列表或numpy数组）
    :return: 行列式值（浮点数）
"""
def calculate_determinant(matrix):

    # 1.将输入统一转换为numpy数组（兼容普通列表输入）
    matrix = np.array(matrix, dtype=np.float64)


    # 2.检验是否是二维方阵（只有方阵才有行列式）
    if matrix.ndim != 2:
        raise ValueError("输入必须是二维矩阵！")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("只有方阵才能计算行列式（行数必须等于列数）！")


    # 3.numpy内置函数计算行列式值
    det_value = np.linalg.det(matrix)
    return det_value



# 示例演示
matrix_2d = [
    [1, 2],
    [3, 4]
]
det_2d = calculate_determinant(matrix_2d)
# 1 * 4 - 2 * 3 = -2
print(f'2阶矩阵行列式：{det_2d:.2f}')                        # 2阶矩阵行列式：-2.00

print('-' * 25)

det_2d_T = np.array(matrix_2d, dtype=np.float64).T
print(f'2阶矩阵的转置行列式：{np.linalg.det(det_2d_T)}')      # 2阶矩阵的转置行列式：-2.0

matrix_3d = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
det_3d = calculate_determinant(matrix_3d)
# 1 * 5 * 9 + 2 * 6 * 7 + 4 * 8 * 3 - 3 * 5 * 7 - 2 * 4 * 9 - 6 * 8 * 1 = 0
print(f'\n3阶矩阵行列式：{det_3d:.2f}')      # 3阶矩阵行列式：0.00

print('=' * 30)

matrix = [
    [2, 3, 5],
    [7, 11, 13]
]
# 抛出异常：ValueError: 只有方阵才能计算行列式（行数必须等于列数）！
calculate_determinant(matrix)

matrix = [
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
]
# 抛出异常：ValueError: 输入必须是二维矩阵！
calculate_determinant(matrix)

