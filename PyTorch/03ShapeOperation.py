'''
形状操作
'''
import torch
import numpy as np


# 创建2x3x4的张量
tensor = torch.randn(2, 3, 4)
print('原始形状:', tensor.shape)         # torch.Size([2, 3, 4])
print('维度:', tensor.dim())            # 3
print("各维度大小:", tensor.size())      # torch.Size([2, 3, 4])
print("特定维度大小:", tensor.size(1))    # 3


print('\n' + '=' * 40)


# reshape和view功能相似，但view有更严格的限制
'''
# **重要区别：**
# 1.view要求张量在内存中是连续的
# 2.reshape会自动处理连续性，更安全
'''
# 0-11, shape: (12,)
tensor = torch.arange(12)

# 重塑为3×4
reshaped = tensor.reshape(3, 4)
print('\nreshape(3, 4):', reshaped.shape)   # torch.Size([3, 4])

# 也可以使用view
viewd = tensor.view(3, 4)
print('view(3, 4):', viewd.shape)           # torch.Size([3, 4])

# 自动推算维度
auto = tensor.reshape(-1, 3)                # -1表示自动计算
print("reshape(-1, 3):", auto.shape)        # torch.Size([4, 3])


print('\n' + '=' * 40)


tensor = torch.arange(1, 10).reshape(3, 3)
'''
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
'''
print(tensor)
print(tensor.dim())                 # 2
# 展平为1维
print(tensor.reshape(-1))           # tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(tensor.reshape(-1).dim())     # 1


print('\n' + '=' * 40)


# 转置
tensor = torch.randn(2, 3, 4, 5)
print(f'\n形状: {tensor.shape}')       # torch.Size([2, 3, 4, 5])
print(f'维度: {tensor.dim()}')         # 4

# 交换第0维和第1维
transposed_0_1 = tensor.transpose(0, 1)
print(f'transpose(0, 1): {transposed_0_1.shape}')    # torch.Size([3, 2, 4, 5])

# 交换最后两个维度
transpose_last = tensor.transpose(-2, -1)
print(f'transpose(-2, -1): {transpose_last.shape}')  # torch.Size([2, 3, 5, 4])
# tensor.transpose(-1, -2) <=> tensor.transpose(-2, -1)
print(np.allclose(tensor.transpose(-1, -2), tensor.transpose(-2, -1)))  # True

