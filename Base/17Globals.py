# 基础用法
x = 10
y = 'Hello'

g = globals()

print(g["x"])   # 10
print('y' in g) # True
print('z' in g) # False


print('-' * 30)


# 动态读取、修改、新增、删除
a = 1

# 读取
print(globals()['a'])   # 1

# 修改
globals()['a'] = 10
print(a)    # 10

# 新增
globals()['b'] = 2
print(globals()['b'])    # 2

# 删除
del globals()['b']
try:
    if 'b' not in globals():
        raise ValueError('b不存在！')
    print(globals()['b'])
except ValueError as e:
    print(e)


print('-' * 30)


# 方法中使用
def globals_value(key: str):
    '''
    获取全局变量的值
    :param key: 变量名
    :return: 变量值
    '''

    if key not in globals():
        raise ValueError(f'全局变量中{key}不存在！')
    return globals().get(key)

# 测试
try:
    globals_value('b')
except ValueError as e:
    print(e)
