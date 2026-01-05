'''
二维数组元素提取
'''
import numpy as np


a = np.arange(9).reshape(3, 3)
'''
[[0 1 2]
 [3 4 5]
 [6 7 8]]
'''
print(a)


'''
提取为一维数组
'''
b1 = a[:, 0]
print(f'\nb1形状:{b1.shape}')     # b1形状:(3,)
'''
[0 3 6]
'''
print(f'a所有行，第0列:{b1}')         # a所有行，第0列:[0 3 6]

b2 = a[:, 0]
print(f'\nb2形状:{b2.shape}')     # b1形状:(3,)
'''
[3 4 5]
'''
print('a所有列，第1行:%s' % str(b2))  # a所有列，第1行:[0 3 6]


print('\n' + '-' * 30)


'''
提取为二维数组
'''
c1 = a[:, [0]]
print(f'\nc1形状:{c1.shape}')     # c1形状:(3, 1)
'''
c1:
[[0]
 [3]
 [6]]
'''
print(f'c1:\n{c1}')

c2 = a[[1], :]
print(f'\nc2形状:{c2.shape}')     # c2形状:(1, 3)
'''
c2:
[[3 4 5]]
'''
print(f'c2:\n{c2}')