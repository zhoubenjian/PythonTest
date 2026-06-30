# Lambda表达式实现加法
add_lambda = lambda a, b: a + b
print(add_lambda(1, 2))


print('-' * 50)


'''
场景使用
'''
# 1.配合sort()排序
students = [('张三', 18), ('李四', 17), ('王五', 15)]
# 按年龄顺序排序
students.sort(key=lambda x: x[1])
print(students)     # [('王五', 15), ('李四', 17), ('张三', 18)]

print('*' * 30)

# 2.配合map()批处理，实现元素的平方
nums = [2, 3, 5, 7]
squarred = list(map(lambda x: x ** 2, nums))
print(squarred)     # [4, 9, 25, 49]

print('*' * 30)

# 3.配合filter()过滤偶数元素
nums = [1, 2, 3, 4, 5, 6, 7]
even = list(filter(lambda x: x % 2 == 0, nums))
print(even)         # [2, 4, 6]

print('*' * 30)

# 4.配合reduce()实现累乘
from functools import reduce
nums = [1, 2, 3, 4, 5]
result = reduce(lambda x, y: x * y, nums)
print(result)       # 120




