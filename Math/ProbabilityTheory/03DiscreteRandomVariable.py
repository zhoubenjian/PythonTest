'''
离散随机变量（Discrete Random Variable）
'''
import numpy as np


# 掷骰子 离散分布的概率函数PMF（Probability Mass Function）
rolls = np.random.randint(1, 7, size=10000)
for v in range(1, 7):
    print(f'掷出{v}的概率：{np.mean(rolls == v):.4f}')


print('\n' + '-' * 30 + '\n')


# 伯努利分布 离散分布的概率函数PMF（Probability Mass Function）
bernoulli = np.random.binomial(1, 0.3, size=10000)
print('伯努利分布：', f'{np.mean(bernoulli):.2f}', sep='')


print('\n' + '-' * 30 + '\n')


# 二项分布 Binomial(10, 0.3)
samples = np.random.binomial(10, 0.3, size=10000)
k_best = np.bincount(samples).argmax()
print(f"二项分布 n=10,p=0.3: 最可能 k={k_best} (理论=np=3)")


