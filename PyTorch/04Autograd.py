'''
自动求导
'''
import torch


# 创建一个 0 维张量（标量），值为 2.0（浮点数，只有浮点数张量才能计算梯度）
x = torch.tensor(2.0, requires_grad=True)   # requires_grad=True：需要追踪这个张量的所有运算，为后续反向传播计算梯度做准备
y = torch.tensor(3.0, requires_grad=True)

# 定义函数：z = x^3 + 2y^2 + xy + 3x + 2y - 10
z = x ** 3 + 2 * y ** 2 + x * y + 3 * x + 2 * y - 10

# 反向传播
z.backward()

# 对x求偏导(y常数)：dz/dx = 3x^2 + y + 3
print(x.grad)   # tensor(18.)
# 对y求偏导(x常数)：dz/dy = 4y + x + 2
print(y.grad)   # tensor(16.)


