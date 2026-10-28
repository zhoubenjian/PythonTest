# * 捕获剩余元素
# 一个赋值左边最多只能出现一次 *，它拿到的永远是 list。
first, *rest = [2, 3, 5, 7, 11, 13, 17, 19]
print(first)    # 2
print(rest)     # [3, 5, 7, 11, 13, 17, 19]

print()

first, *mid, last = [2, 3, 5, 7, 11, 13, 17, 19]
print(first)    # 2
print(mid)      # [3, 5, 7, 11, 13, 17]
print(last)     # 19


print('-' * 20)


'''
* 拆位置参数
'''
def add(a, b, c):
    return a + b + c

nums = [1, 2, 3]
# add(*nums) <=> add(1, 2, 3)
print(add(*nums))   # 6


print('-' * 20)


'''
** 拆关键字参数
'''
def greet(name, age, **kwargs):
    '''
    欢迎函数
    :param name: 姓名
    :param age: 年龄
    :param kwargs: 其他关键字参数（避免报错，原则上字典与函数参数全匹配）
    :return:
    '''
    print(f"Hello, {name}! I am {age} years old.")

user_info = {
    "name": "Alex",
    "age": 30,
    "email": "alex@example.com",
    "phone": "123-456-7890",
    "address": "123 Main St, Anytown, USA",
}
greet(**user_info)  # Hello, Alex! I am 30 years old.



