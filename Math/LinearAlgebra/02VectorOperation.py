'''
矩阵运算
'''
import numpy as np


a = np.array([2, 3])
b = np.array([5, 7])

# 向量加法
print(f'a + b = {a + b}')       # [7 10]

# 向量减法
print(f'a - b = {a - b}')       # [-3 -4]

# 向量数乘
print(f'2.5 * a = {2.5 * a}')   # [5.  7.5]

# 相反（负）向量
print(f'-a = {-a}')             # [-2, -3]