import numpy as np



print(np.arange(5))         # [0 1 2 3 4]
print(np.arange(0, 5, 2))   # [0, 2, 4]

# 一维数组的点积（向量点积）
arr1 = np.ones(5)           # [1. 1. 1. 1. 1.]
arr2 = np.arange(5)         # [0 1 2 3 4]
print(f'向量点积：{np.dot(arr1, arr2)}')     # 10.0
# 使用 @ 运算符（Python 3.5+）
print('向量点积：%f' % (arr1 @ arr2))        # 10.000000

print('\n**********************\n')

# 二维数组的点积（矩阵乘法）
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

b = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

'''
[[22 28]
 [49 64]]
'''
# dot
print(f'矩阵点积：\n{np.dot(a, b)}')
# 使用 @ 运算符
print('矩阵点积：\n%s' % (a @ b))
# matmul
print('矩阵点积：\n', np.matmul(a, b))

print('\n**********************\n')

# 创建两个相同形状的矩阵（哈达玛积）
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
'''
哈达玛积：
[[ 5 12]
 [21 32]]
'''
print(f'哈达玛积：\n{np.multiply(A, B)}')