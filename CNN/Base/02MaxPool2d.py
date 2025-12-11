'''
最大池化操作
'''
import torch
import torch.nn.functional as F


# 直接创建PyTorch tensor
A_tensor = torch.randint(1, 10, (4, 4), dtype=torch.float32)
print("原始tensor A：")
print(A_tensor)

# 添加batch和channel维度
A_tensor = A_tensor.view(1, 1, 4, 4)    # 使用view改变形状

# 执行最大池化（2x2窗口，步长为2）
max_pool2d_result = F.max_pool2d(A_tensor, kernel_size=2)
print("\n最大池化结果：")
print(max_pool2d_result[0, 0])          # 去掉batch和channel维度


