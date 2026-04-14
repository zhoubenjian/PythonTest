'''
模拟梯度下降算法
'''
import numpy as np


# 1.定义目标函数和它的梯度（导数）
def cost_function(x):
    return 2 * x ** 2 - 4 * x + 10


# 目标函数的梯度函数（导函数）
def gradient(x):
    return 4 * x - 4


# 2.实现梯度下降算法
def gradient_descent(start_x, learning_rate, epochs):

    x = start_x
    history = [x]                   # 记录每一步x的位置，用户可视化

    for i in range(epochs):

        grad = gradient(x)

        # 核心更新公式：向梯度反方向走一步
        x -= learning_rate * grad
        history.append(x)

        # 如果梯度非常接近0，说明已经到达最低点，可以提前停止（经常用于实际应用：判断梯度是否为0，而不是判断梯度的绝对值是否小于diff）
        if np.abs(grad) < 1e-6:
            print(f"在第 {i + 1} 次迭代时收敛，x ≈ {x:.6f}")
            break

    return x, cost_function(x)


# 3.运行算法
# 从一个离目标较远的点 x=10 开始，学习率为0.1
final_x, min_value = gradient_descent(start_x=10, learning_rate=0.1, epochs=50)
print(f"找到的最小值点 x ≈ {final_x:.4f}, 代价函数最小值 ≈ {min_value:.4f}")
