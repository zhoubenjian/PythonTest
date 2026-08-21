'''
梯度（Gradient）
'''
import numpy as np


# 碗形函数
def f(x, y):
    return x ** 2 + y ** 2


# 梯度
def grad(x, y):
    return np.array([2 * x, 2 * y])


pt = np.array([1.5, 1.0])


g = grad(*pt)
print(f"在 (1.5, 1.0): 梯度 = {g}")
print(f"梯度模长 = {np.linalg.norm(g):.2f}")
print(f"梯度指向远离原点（上升），负梯度指向原点（下降）")
