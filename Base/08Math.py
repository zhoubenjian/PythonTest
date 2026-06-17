import math

a = 10
b = 3
print(a / b)            # 3.3333333333333335
print(a // b)           # 3
print(a * b)            # 30
print(a ** 10)          # 1000


print('------------------------------------')


print('%d' % (math.log2(1024)))     # 10

# e为底的自然对数
print(math.log(100))    # 4.605170185988092

# 10为底的常用对数
print(math.log10(99))   # 1.99563519459755


print('------------------------------------')


result = math.sin(math.pi / 6)
print(result)           # 0.49999999999999994
print(result == 0.5)    # False
print(math.isclose(result, 0.5))    # True

print(math.sin(math.pi / 4) == math.cos(math.pi / 4))   # True