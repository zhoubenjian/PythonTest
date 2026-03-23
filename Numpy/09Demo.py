import numpy as np


'''
reshape
'''
array01 = np.arange(4).reshape(2, 2)
'''
[[0 1]
 [2 3]]
'''
print(array01)

'''
行向量
'''
array02 = array01.reshape(1, -1)
print('行向量：%s' % array02)           # 行向量：[[0 1 2 3]]
print(f'行向量形状：{array02.shape}')     # 行向量形状：(1, 4)

'''
列向量
'''
array03 = array01.reshape(-1, 1)
'''
列向量：
[[0]
 [1]
 [2]
 [3]]
'''
print('列向量：\n%s' % array03)
print(f'列向量形状：{array03.shape}')     # 列向量形状：(4, 1)


print('----------------------------')


'''
单位矩阵
'''
eye_array = np.eye(5)
'''
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''
print(eye_array)