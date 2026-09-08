import numpy as np
import torch


'''
np.clip：限高，防坠函数
    np.clip(a, a_min, a_max) 的作用就是把数组 a 中所有小于 a_min 的值变成 a_min，所有大于 a_max 的值变成 a_max，中间的不变
'''
arr = np.array([-5, 1, 3, 10])
clipped = np.clip(arr, 0, 5)
print(clipped)      # [0 1 3 5]


'''
ReLU = max(0, x)
'''
a = torch.tensor([1.5, -0.8, 1.1, -0.9, 0.9])
b = torch.relu(a)
print(b)    # tensor([1.5000, 0.0000, 1.1000, 0.0000, 0.9000])

'''
Sigmoid = 1 / (1 + exp(-x))
'''
c = torch.sigmoid(a)
print(c)    # tensor([0.8176, 0.3100, 0.7503, 0.2891, 0.7109])


print('-' * 55)


# 3 x 4 矩阵
arr = np.array([
    [2, 3, 5, 7],
    [11, 13, 17, 19],
    [23, 29, 31, 37]
])

# 每列(axis=0)的最大值
row_max_value = np.max(arr, axis=0, keepdims=True)
print(row_max_value)        # [[23 29 31 37]]
print(np.max(arr, axis=0, keepdims=False))      # [23 29 31 37]


print("-" * 30)


# 每行(axis=1)的最大值
column_max_value = np.max(arr, axis=1, keepdims=True)
'''
[[ 7]
 [19]
 [37]]
'''
print(column_max_value)
print(np.max(arr, axis=1, keepdims=False))      # [ 7 19 37]


score = np.array([0.0, 1.0, 2.0])
max_value = np.max(score, axis=-1, keepdims=True)
shifted = score - max_value
print(shifted)
