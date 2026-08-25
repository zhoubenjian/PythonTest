'''
数学函数
'''
import numpy as np


print(f'自然常数e：{np.e}')      # 自然常数e：2.718281828459045
print(f'圆周率π：{np.pi}')       # 圆周率π：3.141592653589793

# np.log(x)：自然对数，底数e，lnx
print(np.log(np.e))     # 1.0

# 信息论 / 熵用这个
print(np.log2(1024))    # 10.0

# 常用对数，底数为10，常用在统计和工程中，lgx
print(np.log10(100))    # 2.0

# e^x，自然指数；log 的逆运算
print(np.exp(2))        # 7.38905609893065

# 2^x，2的指数；log2 的逆运算
print(np.exp2(10))      # 1024.0


print('\n' + '-' * 50 + '\n')


# (算数)平方根
print(np.sqrt(9))       # 3.0

# 平方
print(np.square(9))     # 81

# a^b，a的b次方
print(np.power(3, 3))   # 27

# 立方根
print(np.cbrt(-729))    # 9.0


print('\n' + '-' * 50 + '\n')


# 角度转换为弧度
print(np.deg2rad(90))       # 1.5707963267948966

# 弧度转换为角度
print(np.rad2deg(np.deg2rad(90)))   # 90.0

# 正弦(弧度制)
print(np.sin(np.pi / 4))    # 0.7071067811865476
