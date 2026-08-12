'''
梯度下降算法
'''
# 模拟代价函数
def cost(x):
    return 2 * x ** 2 - 4 * x + 10


# 代价函数梯度函数
def gradient_cost(x):
    return 4 * x - 4


# 迭代（梯度下降）
def gradient_descent(alpha, max_iters, diff):
    # 初始位置
    x = 10
    for i in range(1, max_iters + 1):

        gradient = gradient_cost(x)

        # 核心更新公式：通过学习率向梯度反方向移动
        new_x = x - (gradient * alpha)

        # 如果参数变化量很小，说明已经收敛
        if abs(x - new_x) <= diff:
            print(f'共迭代了：{i}次')
            break
        x = new_x
    return x, cost(x)


if __name__ == '__main__':
    alpha = 0.1         # 学习率
    max_iters = 100     # 最大迭代次数
    diff = 1e-10        # 误差范围
    x_opt, cost_opt = gradient_descent(alpha, max_iters, diff)
    print('x ≈ %.5f，代价函数最小值 ≈ %.5f' % (x_opt, cost_opt))
