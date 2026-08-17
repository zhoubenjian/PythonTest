import numpy as np


# 标量
scalar = np.array(1)

# 向量（1维张量）
vector = np.array([2, 3, 5, 7, 11, 13, 17, 19])

# 矩阵（二维张量）
matrix = np.array([
    [23, 29, 31],
    [37, 41, 43]
])

# 3D张量(层数, 行数, 列数)
tensor_3d = np.array([
    [[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10, 11, 12]]
])

# 4D张量(模拟3张 2x2 像素的单通道图片)
batch_images = np.array([
    [[[10], [20]], [[30], [40]]],
    [[[50], [60]], [[70], [80]]],
    [[[90], [100]], [[110], [120]]]
])


print("标量:    shape=", scalar.shape, "   ndim=", scalar.ndim)               # shape= ()    ndim= 0
print("向量:    shape=", vector.shape, "   ndim=", vector.ndim)               # shape= (8,)    ndim= 1
print("矩阵:    shape=", matrix.shape, "   ndim=", matrix.ndim)               # shape= (2, 3)    ndim= 2
print("3D 张量: shape=", tensor_3d.shape, "  ndim=", tensor_3d.ndim)          # shape= (2, 2, 3)   ndim= 3
print("4D 张量: shape=", batch_images.shape, " ndim=", batch_images.ndim)     # shape= (3, 2, 2, 1)  ndim= 4


print(tensor_3d[1, :, :])
