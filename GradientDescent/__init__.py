'''
模拟梯度下降算法
'''
import numpy as np


# 1.定义目标函数和它的梯度（导数）
def cost_function(x):
    return x ** 2 + 5 * x + 6


def gradient(x):
    return 2 * x + 5


# 2.实现梯度下降算法
def gradient_descent(start_x, learning_rate, epochs):

    x = start_x
    history = [x]                   # 记录每一步x的位置，用户可视化

    for i in range(epochs):
        grad = gradient(x)
        x -= learning_rate * grad   # 核心更新公式：向梯度反方向走一步
        history.append(x)

        # 如果梯度非常接近0，说明已经到达最低点，可以提前停止
        if np.abs(grad) < 1e-6:
            print(f"在第 {i + 1} 次迭代时收敛，x ≈ {x:.6f}")
            break

    return x, history


# 3.运行算法
# 从一个离目标较远的点 x=10 开始，学习率为0.1
final_x, path = gradient_descent(start_x=10, learning_rate=0.1, epochs=50)
print(f"找到的最小值点 x ≈ {final_x:.4f}")
