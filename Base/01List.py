list1 = [2, 3, 5, 7]
list2 = [11, 13, 17, 19]
list3 = list1 + list2

# 合并
print(list3)            # [2, 3, 5, 7, 11, 13, 17, 19]

# 扩展
list3.extend([23, 29, 31])
print(list3)            # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

# 检查
print(3 in list3)       # True
print(4 in list3)       # False

# 查找（返回索引）
print(list3.index(7))   # 3

# 排序（倒序）（改变原列表）
list3.sort(reverse=True)
print(list3)            # [31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
# 排序（顺序）（改变原列表）
list3.sort()
print(list3)            # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]