'''
sign函数, zip用法, enumerate用法
'''
import numpy as np


a = np.array([-1, -3, 2, -7, 0, 5])
# 负数 -> -1; 正数 -> 1；零 -> 0
print(np.sign(a))   # [-1 -1  1 -1  0  1]

print('\n' + '=' * 30)

X = np.array([2, 3, 5, 7, 11, 13, 17, 19])
print(X.shape)      # (8,)
y = np.arange(1, 9).reshape(8, -1)
print(y.shape)      # (8, 1)

print('\n' + '#' * 30)

for i, (xi, yi) in enumerate(zip(X, y)):
    print(f'{i + 1}: {xi:2d} -> {yi}')
