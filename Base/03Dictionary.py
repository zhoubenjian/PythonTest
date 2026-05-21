'''
字典（Dictionary）{}
'''

person = {'name': 'Michael', 'gender': 'male', 'age': 31, 'city': 'Chicago'}

# 遍历字典的键
for key in person.keys():
    print(key)

print('----------------')

# 遍历字典的值
for value in person.values():
    print(value)


print()


'''
QUERY
'''
print(person.get('name'))                               # Michael

# 键不存在时，返回默认值
print(person.get('universe', 'Harvard University'))     # Harvard University


print()


'''
MODOFY
'''
person['age'] = 33
person.update({'city': 'Miami', 'universe': 'Harvard University'})
print(person)           # {'name': 'Michael', 'gender': 'male', 'age': 33, 'city': 'Miami', 'universe': 'Harvard University'}


print()


'''
DELETE
'''
# 方式1：使用 del 关键字
del person["city"]

# 方式2：使用 pop() 方法（删除并返回该值）
print(person.pop("age"))    # 33

# 方式3：使用 popitem() 删除最后一个插入的键值对（Python 3.7+）
last_item = person.popitem()
print(last_item)            # ('universe', 'Harvard University')
