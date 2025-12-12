'''
最大池化层
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# 创建模拟图像数据 (batch_size, channels, height, width)
batch_size = 1
channels = 1
height, width = 6, 6


# 创建模拟图像
image = torch.tensor([[
    [[1, 2, 3, 4, 5, 6],
     [7, 8, 9, 10, 11, 12],
     [13, 14, 15, 16, 17, 18],
     [19, 20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29, 30],
     [31, 32, 33, 34, 35, 36]]
]], dtype=torch.float32)

print("原始图像尺寸:", image.shape)
print(f"原始图像数据:\n{image.squeeze()}")


print('\n' + '=' * 40)


# 使用nn.MaxPool2d
maxpool2d = nn.MaxPool2d(2, 2)      # 池化核2x2，步长2
output1 = maxpool2d(image)
print("\n使用nn.MaxPool2d (2x2, stride=2):")
print("输出尺寸:", output1.shape)
'''
tensor([[ 8., 10., 12.],
        [20., 22., 24.],
        [32., 34., 36.]])
'''
print(f"输出数据:\n{output1.squeeze()}")


print('\n' + '=' * 40)


# 使用F.max_pool2d
output2 = F.max_pool2d(image, 2, 2) # 池化核2x2，步长2
print("\n使用F.max_pool2d (2x2, stride=2):")
'''
tensor([[ 8., 10., 12.],
        [20., 22., 24.],
        [32., 34., 36.]])
'''
print(f"输出数据:\n{output2.squeeze()}")

# 不同参数的最大池化
print("\n------- 不同参数的最大池化 -------")

# 池化核3x3，步长1
maxpool_3x3 = nn.MaxPool2d(3, 1)
output3 = maxpool_3x3(image)
print("\n3x3池化，步长为1:")
print("输出尺寸:", output3.shape)
'''
tensor([[15., 16., 17., 18.],
        [21., 22., 23., 24.],
        [27., 28., 29., 30.],
        [33., 34., 35., 36.]])
'''
print(f"输出数据:\n{output3.squeeze()}")

# 2x2池化，步长为2，填充1
maxpool_padding = nn.MaxPool2d(2, 2, 1)
output4 = maxpool_padding(image)
print("\n2x2池化，步长为2，填充1:")
print("输出尺寸:", output4.shape)
'''
tensor([[ 1.,  3.,  5.,  6.],
        [13., 15., 17., 18.],
        [25., 27., 29., 30.],
        [31., 33., 35., 36.]])
'''
print(f"输出数据:\n{output4.squeeze()}")
