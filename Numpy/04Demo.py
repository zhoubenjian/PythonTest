import numpy as np


# 标量（0D）
arr0 = np.array(2)
print('标量(0D): ' + str(arr0))       # 标量(0D): 2


print("\n"+"=" * 30)


# 向量（1D）
arr1 = np.array([3, 5])
print('\n向量(1D): %s' % arr1)        # 向量(1D): [3 5]

# 整形数组插入浮点数，浮点数会被截断：[3 5]
arr1[0] = 3.14
print('向量(1D): %s' % arr1)          # 向量(1D): [3 5]

# 一个元素是浮点型，整个向量转为浮点型
arr1 = np.array([3, 5.])
print('浮点向量(1D): ' + str(arr1))    # 浮点向量(1D): [3. 5.]


print("\n" + "=" * 30)


# 矩阵（2D）
arr2 = np.array([[2, 3], [5, 7]])
'''
[[2 3]
 [5 7]]
'''
print(f'\n矩阵(2D):\n{arr2}')


print("\n" + "=" * 30)


# 张量（3D）
arr3 = np.array([[[2, 3], [5, 7], [11, 13]]])
'''
[[[ 2  3]
  [ 5  7]
  [11 13]]]
'''
print(f'\n张量(3D):\n{arr3}')