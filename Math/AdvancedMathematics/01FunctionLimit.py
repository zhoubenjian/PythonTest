'''
函数极限
'''
import numpy as np


def f(x):
    return (x ** 2 - 1) / (x - 1)


print("------- lim_{x->1} (x^2-1)/(x-1) = 2 -------\n")


# 从左右两边逼近
for delta in [0.1, 0.01, 0.001, 0.0001]:
    print(f'x = {(1 - delta):.4f}, f(x) = {f(1 - delta):.6f}')
    print(f'x = {(1 + delta):.4f}, f(x) = {f(1 + delta):.6f}')
    print('-' * 30)
print('\n无论从左右两边逼近，都趋于2。')
