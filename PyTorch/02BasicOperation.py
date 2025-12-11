'''
基础运算
'''
import torch


# 一维张量(向量)
a = torch.tensor([2, 3, 5])
b = torch.tensor([7, 11, 13])

# 加法
print(f'a + b = {a + b}')       # a + b = tensor([ 9, 14, 18])

# 减法
print(f'a - b = {a - b}')       # a - b = tensor([-5, -8, -8])

# 乘法(逐元素相乘)
print(f'a * b = {a * b}')       # a * b = tensor([14, 33, 65])

# 乘法(点积)
print(f'a @ b = {a @ b}')               # a @ b = 112
print(f'a @ b = {torch.matmul(a, b)}')  # a @ b = 112


print('\n' + "#" * 50)


# 二维张量(矩阵)
c = torch.tensor([
    [2, 3],
    [5, 7]
])
d = torch.tensor([
    [11, 13],
    [17, 19]
])

# 加法
'''
tensor([[13, 16],
        [22, 26]])
'''
print(f'\nc + d = {c + d}')

# 减法
'''
tensor([[ -9, -10],
        [-12, -12]])
'''
print(f'c - d = {c - d}')

# 乘法(逐元素相乘)
'''
tensor([[ 22,  39],
        [ 85, 133]])
'''
print(f'c * d = {c * d}')

# 乘法(点积)
'''
tensor([[ 73,  83],
        [174, 198]])
'''
print(f'c @ d = {c @ d}')
'''
tensor([[ 73,  83],
        [174, 198]])
'''
print(f'c @ d = {torch.matmul(c, d)}')
