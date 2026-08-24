'''
图像矩阵
生成一张合成灰度图，用 NumPy 数组操作实现转置、翻转、裁剪和手写卷积模糊。
'''
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# -------------------------- 设置中文字体 start --------------------------
plt.rcParams['font.sans-serif'] = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end -------------------------


# 创建输出目录
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


# 1. 生成一张合成灰度图：对角渐变 + 中间亮方块
size = 64
img = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        # 从暗到亮的对角渐变
        img[i, j] = (i + j) / (2 * size)

# 中间加一个亮方块
img[20:44, 20:44] += 0.6

# 限制在 [0, 1] 范围内
img = np.clip(img, 0, 1)

print("图像矩阵形状 (shape):", img.shape, sep='')
print("图像矩阵前 3x3 部分：\n", np.round(img[:3, :3], 3), sep='')




