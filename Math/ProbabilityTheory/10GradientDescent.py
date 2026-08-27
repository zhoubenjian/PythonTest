'''
梯度下降(Gradient Descent)
'''
import numpy as np


# 生成数据 y = 3x + 2 + noise
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 3 * X + 2 + np.random.normal(0, 2, 100)


# 从零开始实现梯度下降
w, b = 0.0, 0.0
lr, iterate = 0.01, 1000
losses = []


for i in range(iterate):
    y_pred = w * X + b
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)

    '''
    梯度
    '''
    # 损失函数对w偏导数
    dw = 2 * np.mean((y_pred - y) * X)
    # 损失函数对b偏导数
    db = 2 * np.mean(y_pred - y)
    # 更新参数
    w -= lr * dw
    b -= lr * db

print(f"梯度下降结果:")
print(f"真实: w=3.0, b=2.0")
print(f"拟合: w={w:.4f}, b={b:.4f}")
print(f"初始损失: {losses[0]:.2f} => 最终损失: {losses[-1]:.2f}")

