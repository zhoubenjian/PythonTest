import numpy as np


'''
np.dot（内积/点积）
'''
# 一维数组
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
print('a和b的内积：', np.dot(a, b), sep='')      # 20
'''
[[ 2  3  4]
 [ 4  6  8]
 [ 6  9 12]]
'''
print('a和b的外积：\n', np.outer(a, b), sep='')

print('\n' + '-' * 50 + '\n')

# 二维矩阵
c = np.array([
    [1, 2],
    [3, 4]
])
d = np.array([
    [2, 3],
    [4, 5]
])
'''
[[10 13]
 [22 29]]
'''
print('c和d的内积：\n', np.dot(c, d), sep='')
# 会将c，d展平为一维数组（外积仅支持一维数组），再计算外积，结果为二维矩阵
'''
[[ 2  3  4  5]
 [ 4  6  8 10]
 [ 6  9 12 15]
 [ 8 12 16 20]]
'''
print('c和d的外积：\n', np.outer(c, d), sep='')
