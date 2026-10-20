'''
normal(μ, σ) = μ + σ · randn()
'''
import numpy as np


# 正态分布（均值：5，标准差：2）
data = np.random.normal(5, 2, 10)
print('随机数组:', data, '\n', '均值:', f'{np.mean(data):.3f}', '\n', '方差:', f'{np.var(data):.3f}', '\n', '标准差:', f'{np.std(data):.3f}', sep='')

# 标准正态分布(μ=0, σ²=1)
standard_normal = np.random.randn(10)
print('\n标准正态分布:', standard_normal, '\n', '均值:', f'{np.mean(standard_normal):.3f}', '\n', '方差:', f'{np.var(standard_normal):.3f}', '\n', '标准差:', f'{np.std(standard_normal):.3f}', sep='')


print('\n' + '-' * 50 + '\n')


img = np.arange(16).reshape(4, 4)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
'''
print('原始图像矩阵:\n', img, sep='')

# 水平镜像：列索引反转，行索引保持不变
img_flip_lr = img[:, ::-1]
'''
[[12 13 14 15]
 [ 8  9 10 11]
 [ 4  5  6  7]
 [ 0  1  2  3]]
'''
print('\n水平镜像后的图像矩阵:\n', img_flip_lr, sep='')

# 垂直镜像：行索引反转，列索引保持不变
img_flip_ud = img[::-1, :]
'''
[[15 14 13 12]
 [ 9  8  7  6]
 [ 5  4  3  2]
 [ 1  0  0  0]]
'''
print('\n垂直镜像后的图像矩阵:\n', img_flip_ud, sep='')

# 转置(行列互换)
img_transposed = img.T
'''
[[ 0  4  8 12]
 [ 1  5  9 13]
 [ 2  6 10 14]
 [ 3  7 11 15]]
'''
print('\n转置后的图像矩阵:\n', img_transposed, sep='')


print('\n' + '-' * 50 + '\n')


# 模拟边缘填充
size = 64
img = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        img[i, j] = (i + j) / (2 * size)  # 从暗到亮的对角渐变
img[20:44, 20:44] += 0.6                  # 中间加一个亮方块
img = np.clip(img, 0, 1)                  # 限制在 [0, 1] 范围内
print('原始图像矩阵:\n', img, sep='')

# pad = 5 // 2
# print('\n填充宽度:', pad, sep='')
# padded = np.pad(data, pad, mode='edge')
# out = np.zeros_like(data)


def box_blur(data, k = 5):
    """
    模拟边缘填充
    :param data: 输入数据
    :param k: 模糊半径
    :return: 模糊后的数据
    """
    pad = 5 // 2
    # 边界用边缘值填充（edge padding）
    padded = np.pad(data, pad, mode='edge')
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # 取 k x k 邻域，求均值
            out[i, j] = np.mean(padded[i:pad+i, j:pad+j])
    return out

img_blur = box_blur(img, k=5)
print('\n模糊后的图像矩阵:\n', img_blur, sep='')
