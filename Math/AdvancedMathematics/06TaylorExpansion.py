'''
泰勒展开（Taylor Expansion）
'''
import math
import numpy as np


x = 0.5
true_val = np.sin(x)
print(f'sin({x})的泰勒近似值 vs 真实值：{true_val:.6f}')


# 手动计算前几项泰勒展开
approx = 0
for power in range(1, 20, 2):  # 1, 3, 5, ... 19（共10项）
    coef = (-1) ** ((power - 1) // 2) / math.factorial(power)
    approx += coef * x ** power

err = abs(approx - true_val)
print(f"taylor展开近似值: {approx:.6f}  误差: {err:.2e}")
