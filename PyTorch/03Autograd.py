'''
自动求导
'''
import torch


x = torch.tensor(2.0, requires_grad=True)

# 定义函数：y = 3x^3 + 2x^2 + 3x - 10
y = x ** 3 + 2 * x ** 2 + 3 * x - 10

# 求导 dy/dx
y.backward()

print(x.grad)   # tensor(23.)


