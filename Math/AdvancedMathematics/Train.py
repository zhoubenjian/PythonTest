'''
求取cosx的泰勒展开（前10非0项）
'''
import math
import numpy as np


x = 0.5
true_val = np.cos(x)
print(f'cos({x})的泰勒近似值 vs 真实值：{true_val:.6f}')

# 第0项
approx = 1.0
# 直接遍历幂次：2, 4, 6, 8, ... 20（共10项）
for power in range(2, 21, 2):  # range(start=2, stop=21, step=2)
    coef = (-1) ** (power // 2) / math.factorial(power)
    approx += coef * x ** power

err = abs(approx - true_val)
print(f"取到 x^{power} 项（共10项）: {approx:.8f}  误差: {err:.2e}")








