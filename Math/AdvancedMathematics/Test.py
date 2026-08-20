import math
import sympy as sp


# sp.log() 默认底数为e，即自然对数
print(sp.log(math.e ** 10))

print('e =', math.e)                # 2.718281828459045
print('π =', math.pi)               # 3.141592653589793


def log(x, base = 2):
    return sp.log(x, base)


print(f'{log(1024).evalf():.6f}')   # 10.000000
