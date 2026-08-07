'''
最大池化操作
'''
import torch
import torch.nn.functional as F         # 导入神经网络函数库，max_pool2d在这里


# 创建4行4列张量，数值1~9，数据类型float32
A_tensor = torch.randint(1, 10, (4, 4), dtype=torch.float32)
print("原始tensor A：")
'''
tensor([[7., 4., 5., 5.],
        [5., 1., 8., 5.],
        [4., 5., 5., 7.],
        [2., 4., 8., 6.]])
'''
print(A_tensor)

# 增加batch批次维度、channel通道维度
# 形状变化：(4,4) => (1, 1, 4, 4)  格式：(N, C, H, W)
A_tensor = A_tensor.view(1, 1, 4, 4)    # 使用view改变形状

# 2D最大池化，窗口大小kernel_size=2（2×2）
# stride不指定，默认等于kernel_size=2，窗口移动步长2
max_pool2d_result = F.max_pool2d(A_tensor, kernel_size=2)
print("\n最大池化结果：")
'''
tensor([[8., 4.],
        [7., 9.]])
'''
print(max_pool2d_result[0, 0])          # 去掉batch和channel维度



print('\n' * 3)



print('======================================================')
B_tensor = torch.randint(1, 10, (5, 5), dtype=torch.float32)
print('B_tensor:')
'''
tensor([[6., 4., 3., 3., 2.],
        [4., 9., 5., 2., 5.],
        [6., 2., 4., 4., 5.],
        [6., 9., 4., 1., 2.],
        [9., 8., 9., 7., 6.]])
'''
print(B_tensor)

# 增加batch批次维度、channel通道维度 (N(batch), C(channel), H(height), W(width))
# 形状变化：（5, 5）=> (1, 1, 5, 5)
B_tensor = B_tensor.view(1, 1, 5, 5)

'''
kernel_size = 3
    池化窗口大小：**3×3 滑动窗口**，每次框住 3 行 3 列共 9 个元素，取窗口内**最大值**作为输出点。
    
stride = 1
    滑动步长 = 1：窗口每计算完一次，向右 / 向下只移动**1 个格子**。
    窗口之间**大量重叠**，不像之前 stride=2 互不重叠。
    
padding = 0
    边缘填充：0，**不对原图四周补 0**。窗口只能完全落在原始 5×5 矩阵内部，不能超出边界。
    
dilation = 1
    膨胀系数 = 1，普通池化窗口；
    dilation > 1 才是空洞池化，窗口元素之间会跳过像素。dilation=1 就是连续 3×3 方块。
'''
# 2D最大池化，窗口大小kernel_size=3（3×3）
# stride=1，窗口移动步长1
max_pool2d_result = F.max_pool2d(B_tensor, kernel_size = 3, stride = 1, padding = 0, dilation = 1)
print("\n最大池化结果：")
'''
tensor([[9., 9., 5.],
        [9., 9., 5.],
        [9., 9., 9.]])
'''
print(max_pool2d_result[0, 0])          # 去掉batch和channel维度




