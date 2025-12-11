'''
PyTorch实现简单卷积神经网络（CNN）
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 第一个卷积层，输入通道数为1（灰度图像），输出通道数为32，卷积核大小为3 × 3，步长为1，填充为1。
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # 第二个卷积层，输入通道数为32（灰度图像），输出通道数为64，卷积核大小为3 × 3，步长为1，填充为1。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 第一个全连接层，输入大小为NBSP64 × 7 × 7，输出大小为128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # 第二个全连接层，输入大小为128，输出大小为10（对应10个类别）
        self.fc2 = nn.Linear(128, 10)