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


'''
1.生成一张合成灰度图：对角渐变 + 中间亮方块
'''
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


'''
2.矩阵运算 = 图像变换
'''
# 转置(行列互换)
img_transposed = img.T

# 水平镜像：列索引反转，行索引保持不变
img_flip_lr = img[:, ::-1]

# 垂直镜像：行索引反转，列索引保持不变
img_flip_ud = img[::-1, :]

# 裁剪：切片取子矩阵
img_crop = img[10:54, 10:54]


'''
3.手写卷积模糊（5x5 卷积核，不调 OpenCV）
'''
def img_blur(img, k = 5):
    """
    模拟边缘填充
    :param img: 输入图像
    :param k: 模糊半径
    :return: 模糊后的图像
    """
    pad = 5 // 2
    # 边界用边缘值填充（edge padding）
    padded = np.pad(img, pad, mode='edge')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 取 k x k 邻域，求均值
            out[i, j] = padded[i:i+k, j:j+k].mean()     # np.mean(padded[i:pad+i, j:pad+j])
    return out

# 模糊后的图像矩阵
img_blur = img_blur(img, k=5)
print("模糊后矩阵前 3x3 部分：\n", np.round(img_blur[:3, :3], 3))


# 4. 保存可视化对比图
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
titles = ["原图 (矩阵)", "转置 img.T",
          "水平镜像 img[:, ::-1]",
          "垂直镜像 img[::-1, :]",
          "裁剪 img[10:54, 10:54]",
          "均值模糊 (5x5 卷积)"]
images = [img, img_transposed, img_flip_lr,
          img_flip_ud, img_crop, img_blur]

for ax, title, im in zip(axes.flat, titles, images):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_image_as_matrix.png"), dpi=130)
print(f"\n可视化结果已保存到 {OUT}/01_image_as_matrix.png")

