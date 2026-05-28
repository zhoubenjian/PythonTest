'''
格式化输出
'''


a = 2 ** (1/2)
b = 3 ** (1/2)
c = 5 ** (1/2)
d = 7 ** (1/2)


print('a ≈ %.3f' % a)           # 1.414
print('b ≈ {:.3f}'.format(b))   # 1.732
print(f'c ≈ {c:.3f}')           # 2.236
print('d ≈', f'{d:.3f}')        # 2.646

print('-' * 50)

# /：普通除法
print('10 / 3 =', 10 / 3)       # 3.3333333333333335
# //：整除(只保留整数部分)
print('10 // 3 =', 10 // 3)     # 3
# %：取余(取模)
print('10 % 3 =', 10 % 3)       # 1

print('-' * 50)

# **：乘方
print('10 ** 3 =', 10 ** 3)     # 1000


print('-' * 50)


# 字符串截取（Python 切片规则：开始位置 必须 在 结束位置 的左边，才能取到内容）
str = 'Hello World! The great city Chongqing in the southwest of China.'
print(str[:12])                 # Hello World!
print(str[6:12])                # World!
print(str[-6:])                 # China.
print(str[-36:-27])             # Chongqing
