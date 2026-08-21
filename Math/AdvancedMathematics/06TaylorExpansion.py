'''
泰勒展开（Taylor Expansion）
'''
import math
import numpy as np


x0 = 0.5
true_val = np.sin(x0)
print(f'sin({x0})的泰勒近似值 vs 真实值：{true_val:.4f}')

# 手动计算前几项泰勒展开
approx = 0
for n in range(6):
    if n % 2 == 1:  # 偶数项为0
        coef = (-1) ** ((n - 1) // 2) / math.factorial(n)
        approx += coef * x0 ** n
        err = abs(approx - true_val)
        print(f"{n}项: {approx:.8f}  误差: {err:.2e}")
