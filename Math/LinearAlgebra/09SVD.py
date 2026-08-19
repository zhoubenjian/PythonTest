'''
奇异值分解（Singular Value Decomposition）
    A(m x n) = U(m x m正交矩阵) * S(m x n对角矩阵) * V^T

    U：m x m正交矩阵，列为左奇异向量，两两单位正交
    S：m x n对角矩阵，对角元奇异值满足σ1≥σ2≥...≥σr>0(主对角线元素是奇异值)，剩余位置填 0
    V：n x n正交矩阵，列为右奇异向量，两两单位正交，V^T的行是右奇异向量的转置

    适用于所有矩阵
'''
import numpy as np


a = np.array([
    [1, 0, 0, 0, 2],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0]
])

U, S, VT = np.linalg.svd(a)
# 奇异值S
print('奇异值：', np.round(S, 2), sep='')
# 矩阵a的秩等于奇异值S对角元中非零元素(大于0)的个数
print('a的秩：', np.sum(S > 1e-10), sep='')


print('\n' + '-' * 30 + '\n')


# 图像压缩演示：秩为1的图像
img = np.outer(np.array([1, 2, 3, 2, 1]), np.array([2, 4, 6, 4, 2]))
'''
[[ 2  4  6  4  2]
 [ 4  8 12  8  4]
 [ 6 12 18 12  6]
 [ 4  8 12  8  4]
 [ 2  4  6  4  2]]
'''
print('img:\n', img, sep='')

U_i, S_i, VT_i = np.linalg.svd(img)
print('\nimg奇异值：\n', np.round(S_i, 2), sep='')      # [38.  0.  0.  0.  0.]

# 使用1个奇异值重建
i = 1
img_reconstructed = U_i[:, :i] @ np.diag(S_i[:i]) @ VT_i[:i, :]
print('重建误差：', np.linalg.norm(img - img_reconstructed), sep='')     # 1.0185048308013224e-14




