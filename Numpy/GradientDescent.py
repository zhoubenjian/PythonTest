'''
梯度下降（Gradient Descent）
'''
import numpy as np
import matplotlib.pyplot as plt


# 定义损失函数 L(w) = w^2
def loss(w):
    return w ** 2

# 损失函数梯度 dL/dw = 2*w
def gradient(w):
    return 2 * w


# 梯度下降算法
def gradient_descent(start_w, learn_rate, iterations):
    '''
    梯度下降算法
    :param start_w:     起始位置
    :param learn_rate:  学习率
    :param iterations:  迭代次数
    :return:
    '''
    w = start_w

    # 记录 w 的变化历史
    w_history = [w]

    # 记录损失的变化历史
    loss_history = [loss(w)]

    for i in range(1, iterations + 1):
        # 计算当前点的梯度
        grad = gradient(w)

        # 沿负梯度方向更新参数
        w -= (grad * learn_rate)

        w_history.append(w)
        loss_history.append(loss(w))

    return w_history, loss_history



if __name__ == '__main__':

    # 执行梯度下降：从 w=5 开始，学习率 0.1，迭代 20 次
    w_start = 5.0
    lr = 0.1
    iters = 20
    w_hist, loss_hist = gradient_descent(w_start, lr, iters)

    print(f"初始 w: {w_hist[0]:.4f}, 初始损失: {loss_hist[0]:.4f}")
    print(f"最终 w: {w_hist[-1]:.4f}, 最终损失: {loss_hist[-1]:.4f}")


    # 可视化优化过程
    plt.figure(figsize=(12, 4))

    # 图1：损失函数曲线及优化路径
    plt.subplot(1, 2, 1)
    w_vals = np.linspace(-6, 6, 100)
    plt.plot(w_vals, loss(w_vals), label='L(w) = w²')
    plt.scatter(w_hist, loss_hist, c='red', s=20, label='Gradient Descent Steps')
    plt.plot(w_hist, loss_hist, 'r--', alpha=0.5)
    plt.xlabel('Parameter w')
    plt.ylabel('Loss L(w)')
    plt.title('Gradient Descent on L(w)=w²')
    plt.legend()
    plt.grid(True)

    # 图2：损失值随迭代次数的下降曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss_hist)), loss_hist, 'b-o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Reduction Over Iterations')
    plt.grid(True)

    plt.tight_layout()
    plt.show()



