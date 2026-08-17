'''
拟合
'''
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
# -------------------------- 设置中文字体 start --------------------------


np.random.seed(42)
# 生成等差数列 起始：0；结束：10；个数：20
X = np.linspace(0, 10, 20)
# 真实数据规律（我们不知道）
y_true = np.sin(X)
# 随机噪声
y_noise = np.random.randn(20) * 0.3
# 实际观测到的数据
y = y_true + y_noise


plt.scatter(X, y, label='观测数据 (含噪声)', color='blue', alpha=0.6)
plt.plot(X, y_true, label='真实规律 (y=sin(x))', color='green', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('数据与潜在规律')
plt.legend()
plt.grid(True)
plt.show()