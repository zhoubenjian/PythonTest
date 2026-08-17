'''
矩阵运算
    矩阵加减法：
        必须是同型矩阵A(m,n) +/- B(m,n) = C(m,n)
    矩阵乘法:
        A(m,n) x B(n,p) = C(m,p) 且不满足交换律：AB ≠ BA（通常情况）!!!
'''
import numpy as np


# 销量矩阵
sales = np.array([
    [10, 5, 3],
    [8, 7, 4],
    [12, 4, 6]
])

# 价格矩阵
price = np.array([
    [3, 3.5],
    [5, 5.5],
    [4, 4.0]
])


# 矩阵加法（必须同型矩阵）
bonus = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0]
])
total = sales + bonus
# sep=''用于去除空格
'''
[[11  5  4]
 [ 8  8  4]
 [13  5  6]]
'''
print('sales + bonus:\n', total, sep='')


print('\n' + '-' * 33 + '\n')


# 矩阵乘法 (3x2) @ (2x3) = (3x3)
revenue = sales @ price
'''
[[67.  74.5]
 [75.  82.5]
 [80.  88. ]]
 '''
print(f'revenue:\n{revenue}')

# 手动验证(0,0)位置结果
manual = sales[0, 0] * price[0, 0] + sales[0, 1] * price[1, 0] + sales[0, 2] * price[2, 0]
print(f'manual check：{manual} == {revenue[0, 0]}')