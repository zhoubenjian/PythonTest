'''
线性方程组与矩阵的秩
    秩（MRank）：独立信息的数量
    秩（Rank）= 矩阵中真正「独立」的行数（或列数）
    如果某个方程可以由其他方程乘以系数得到，它就是冗余的——不增加新信息

    满秩：
        唯一解，方程之间无冗余
    欠秩：
        无穷多解，有冗余方程
    矛盾：
        无解，方程之间矛盾
'''
import numpy as np


# 鸡兔同笼（满秩）
a = np.array([
    [1, 1],
    [2, 4]
])

# np.linalg.solve会将b作为列向量处理
b = np.array([10, 28])

x = np.linalg.solve(a, b)
print('解：', x, sep='')                              # [6. 4.]
print('秩：', np.linalg.matrix_rank(a), sep='')       # 2


# 冗余方程
c = np.array([
    [1, 2],
    [2, 4]
])
print('\n冗余方程组的秩：', np.linalg.matrix_rank(c), sep='')       # 1


# 随机大矩阵的秩
big = np.random.rand(100, 50)
print('\n随机大矩阵的秩：', np.linalg.matrix_rank(big), sep='')     # 50
