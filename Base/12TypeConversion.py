'''
类型转换
'''

float_str = '3.1415926'
print(float(float_str))         # 3.1415926
print(int(float(float_str)))    # 3

print('\n' + '-' * 30 + '\n')

# 空、零、None 等为 False，其余为 True
print(bool(0))                  # False
print(bool(None))               # False
print(bool(""))                 # False
