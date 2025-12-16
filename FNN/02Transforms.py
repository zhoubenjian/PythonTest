import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


img = Image.open('./images/2025-10-27.jpg')
print(f'原图大小：{img.size}')                   # 原图大小：(1920, 1200)

# (C, H, W)（通道数、高度、宽度，PyTorch 标准格式），[0,1]范围
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img)
print(f'img_tensor形状：{img_tensor.shape}')    # img_tensor形状：torch.Size([3, 1200, 1920])
print(img_tensor)