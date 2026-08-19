'''
导数（Derivative）
'''
import numpy as np


def numerical_derivative(f, x, delta=1e-4):
    """
    计算函数在指定点的数值导数。
    :param f: 函数
    :param x: 点
    :param delta: 步长
    :return:
    """
    return (f(x + delta) - f(x)) / delta


f = lambda x: x ** 2 - 10 * x + 25
print('f′(2) = ', f'{numerical_derivative(f, 2):.6f}', sep='')      # f′(2) = -5.999900


print('\n' + '=' * 50 + '\n')


# 各点的导数
for x in [1, 3, 5, 7, 9]:
    d = numerical_derivative(f, x)
    dir = '下降' if d < 0 else ('上升' if d > 0 else '水平')
    # print(f"x = {x:.2f}, f′(x) = {d:4.1f}, {dir}")
    print('x = ', f'{x:.2f}, ', 'f′(x) = ', f'{d:4.1f}, ', dir, sep='')
