import torch


# 一维张量：形状 (4,)
a = torch.tensor([2, 3, 5, 7])
print(a)            # tensor([2, 3, 5, 7])
print(a.shape)      # torch.Size([4])
print(a.dim())      # 1

print('\n' + '*' * 50 + '\n')

# 索引0处插入新的维度
b = a.unsqueeze(0)
print(b)            # tensor([[2, 3, 5, 7]])
print(b.shape)      # torch.Size([1, 4])
print(b.dim())      # 2

print('\n' + '*' * 50 + '\n')

c = a.unsqueeze(1)
'''
tensor([[2],
        [3],
        [5],
        [7]])
'''
print(c)
print(c.shape)      # torch.Size([4, 1])
print(c.dim())      # 2


print('\n' * 5)


a = torch.randn(2, 3)
print(a)
print(a.shape)          # torch.Size([2, 3])
print(a.dim())          # 2

print('\n' + '-' * 50 + '\n')

x = a.unsqueeze(0)
print(x)
print(x.shape)          # torch.Size([1, 2, 3])
print(x.dim())          # 3

print('\n' + '-' * 50 + '\n')

y = a.unsqueeze(1)
print(y)
print(y.shape)          # torch.Size([2, 1, 3])
print(y.dim())          # 3

print('\n' + '-' * 50 + '\n')

z = a.unsqueeze(2)
print(z)
print(z.shape)          # torch.Size([2, 3, 1])
print(z.dim())          # 3