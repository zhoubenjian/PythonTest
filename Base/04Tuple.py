'''
元组（Tuple）()
它看作是一个“不可变的列表”（只读列表）。
不可变性（核心）：元组一旦创建，其内部的元素不能被修改、添加或删除
'''

# 方式1：使用圆括号（最常用）
tuple = (2, 3, 'five', 7)
print(tuple)            # (2, 3, 'five', 7)

# 方式2：省略括号（仅用逗号分隔，Python 也能识别为元组）
another_tuple = 'eleven', 13, 'seventeen'
print(another_tuple)    # ('eleven', 13, 'seventeen')


print()


'''
QUERY
'''
fruits = ('apple', 'banana', 'cherry', 'orange', 'peach')
# 元组第一个元素
print(fruits[0])        # apple
# 元组最后一个元素
print(fruits[-1])       # peach


