import numpy as np


'''
np.clip：限高，防坠函数
    np.clip(a, a_min, a_max) 的作用就是把数组 a 中所有小于 a_min 的值变成 a_min，所有大于 a_max 的值变成 a_max，中间的不变
'''
arr = np.array([-5, 1, 3, 10])
clipped = np.clip(arr, 0, 5)
print(clipped)      # [0 1 3 5]
