'''
模拟CNN卷积操作
'''
import torch
import torch.nn.functional as F


# 输入特征图（5 x 5）
input_map = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]

# 卷积核（3 x 3）
kernel = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
]


# 1. 转换为PyTorch张量（需适配框架输入格式：[batch, channel, height, width]）
input_tensor = torch.tensor(input_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 2. 调用PyTorch的卷积函数（F.conv2d，默认Padding=0，Stride=1）
# 注意：F.conv2d实现的是互相关，与我们的模拟逻辑一致
pytorch_result = F.conv2d(input_tensor, kernel_tensor, stride=1, padding=0)

# 3. 转换格式并打印结果
print("=== PyTorch框架CNN卷积结果（3×3）===")
print(pytorch_result.squeeze().int())














