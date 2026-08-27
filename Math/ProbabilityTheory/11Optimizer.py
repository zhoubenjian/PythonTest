'''
常用优化器(Optimizer)
     SGD(随机梯度下降)
        简单，但震荡大
        每次随机抽小批量算梯度
        CV 任务，ResNet 训练

     Momentum(动量SGD)
        加速收敛，减少震荡
        积累历史方向，像滚雪球
        Transformer/BERT/GPT 默认选择

     Adam(自适应矩估计)
        默认首选，大部分任务好用
        动量 + 自适应学习率，每个参数有自己的学习率
        大模型训练的事实标准
'''
import numpy as np


# 定义损失函数
def loss(w, b):
    # 最优点在 (w, b) = (3, -2)
    return (w - 3) ** 2 + 2 * (b + 2) ** 2


# SGD优化器
def sgd(lr=0.1, steps=30):
    w, b = 0.0, 0.0
    losses = []
    for _ in range(steps):
        w -= lr * 2 * (w - 3)
        b -= lr * 4 * (b + 2)
        losses.append(loss(w, b))
    return losses


# Momentum优化器
def momentum(lr=0.1, beta=0.9, steps=30):
    w, b, vw, vb = 0.0, 0.0, 0.0, 0.0
    losses = []
    for _ in range(steps):
        vw = beta * vw + lr * 2 * (w - 3)
        vb = beta * vb + lr * 4 * (b + 2)
        w -= vw
        b -= vb
        losses.append(loss(w, b))
    return losses


print("========== 优化器收敛对比 ===========")
for name, fn in [("SGD", sgd), ("Momentum", momentum)]:
    losses = fn()
    print(f"{name:10s}: 初始损失:{losses[0]:.1f}，最终损失:{losses[-1]:.1f}")
