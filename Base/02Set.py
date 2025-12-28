set1 = {1, 2, 3, 4, 5, 5}
print(f'set1: {set1}')      # set1: {1, 2, 3, 4, 5}

set1.update([6, 7])
print(f'set1: {set1}')      # set1: {1, 2, 3, 4, 5, 6, 7}

# 删除存在的元素（元素不存在，报错）
set1.remove(7)
print(f'set1: {set1}')      # set1: {1, 2, 3, 4, 5, 6}

# 删除元素（元素不存在不报错）
set1.discard(9)
print(f'set1: {set1}')      # set1: {1, 2, 3, 4, 5, 6}

# 清空集合
set1.clear();
print(f'set1: {set1}')      # set1: set()