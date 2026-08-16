import numpy as np
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
# -------------------------- 设置中文字体 end --------------------------


a = np.array([3, 1])
b = np.array([1, 2])
sum = a + b

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：三角形法则
for ax, title in zip(axes, ['三角形法则', '平行四边形法则']):
    ax.set_xlim(-1, 6); ax.set_ylim(-1, 5)
    ax.axhline(y=0, color='gray', lw=0.5); ax.axvline(x=0, color='gray', lw=0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.set_title(title)

# 画三角形法则
ax = axes[0]
ax.arrow(0, 0, a[0], a[1], head_width=0.2, head_length=0.2,
         fc='#e74c3c', ec='#e74c3c', lw=2, label='a')
ax.arrow(a[0], a[1], b[0], b[1],
         head_width=0.2, head_length=0.2, fc='#3498db', ec='#3498db', lw=2, label='b')
ax.arrow(0, 0, sum[0], sum[1], head_width=0.2, head_length=0.2,
         fc='#2ecc71', ec='#2ecc71', lw=2, ls='dashed', label='a+b')
ax.legend()

# 画平行四边形法则
ax = axes[1]
ax.arrow(0, 0, a[0], a[1], head_width=0.2, head_length=0.2,
         fc='#e74c3c', ec='#e74c3c', lw=2, label='a')
ax.arrow(0, 0, b[0], b[1], head_width=0.2, head_length=0.2,
         fc='#3498db', ec='#3498db', lw=2, label='b')
ax.arrow(a[0], a[1], b[0], b[1],
         head_width=0.2, head_length=0.2, fc='#3498db', ec='#3498db', lw=1.5, ls='dotted', alpha=0.6)
ax.arrow(b[0], b[1], a[0], a[1],
         head_width=0.2, head_length=0.2, fc='#e74c3c', ec='#e74c3c', lw=1.5, ls='dotted', alpha=0.6)
ax.arrow(0, 0, sum[0], sum[1], head_width=0.2, head_length=0.2,
         fc='#2ecc71', ec='#2ecc71', lw=2, ls='dashed', label='a+b')
ax.legend()
plt.tight_layout(); plt.show()
