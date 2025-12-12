'''
自动求导
'''
import torch


x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 定义函数：z = x^3 + 2y^2 + xy + 3x + 2y - 10
z = x ** 3 + 2 * y ** 2 + x * y + 3 * x + 2 * y - 10

# 反向传播
z.backward()

# 对x求偏导(y常数)：dz/dx = 3x^2 + y + 3
print(x.grad)   # tensor(18.)
# 对y求偏导(x常数)：dz/dy = 4y + x + 2
print(y.grad)   # tensor(16.)


