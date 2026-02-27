import numpy as np
from scipy import optimize, integrate, linalg, stats    # optimize（优化）、integrate（积分）、stats（统计）

'''
1.求最值
'''
def f1(x):
    return x ** 2 + 10 * np.sin(x)


# 用minimize找最小值（初始猜测值x0=0）
min_value = optimize.minimize(f1, x0=0)

# 输出结果
print(f'最小值对应x值：{min_value.x[0]}')  # -1.3064401160169776
print('函数最小值：' + str(min_value.fun))  # -7.945823375615215
print('是否成功收敛：%s' % min_value.success)  # True


print('-' * 40)


'''
2.求解方程根
'''
def eq(x):
    return x ** 2 - 4


# 求解根（初始x0=1）
root = optimize.root(eq, x0=1)
print(f'方程的根：{root.x[0]:.2f}')


print('-' * 40)


'''
3.求数值积分
'''
def integrand(x):
    return x ** 2

# 数值积分（quad是最常用的单变量积分函数）
integral_result, error = integrate.quad(integrand, 0, 1)
print(f'微分结果：{integral_result}')    # 0.33333333333333337
print(f'积分误差：{error}')              # 3.700743415417189e-15


print('-' * 40)


'''
4.进阶线性代数
'''
# 定义矩阵
A = np.array([
    [1, 2],
    [3, 4]
])

# 1.求矩阵的逆
A_inv = linalg.inv(A)
'''
[[-2.   1. ]
 [ 1.5 -0.5]]
'''
print('矩阵的逆：\n', A_inv)

print('#' * 25)

# 2.求行列式的值
det_A = linalg.det(A)
print(f'A矩阵的行列式值：{det_A}')      # -2

print('#' * 25)

# 3.求特征值和特征向量
eigen_values, eigen_vectors = linalg.eig(A)
print(f'特征值：{eigen_values}')      # [-0.37228132+0.j  5.37228132+0.j]


print('-' * 40)


'''
4.统计模块：概率分布与统计分析
'''
# 1.创建标准正态分布（均值0，标准差1）
norm_dist = stats.norm(loc=0, scale=1)

# 2.生成1000个符合该分布的随机数
random_data = norm_dist.rvs(size=1000)

# 3.计算统计量
# 输出≈0（随机，接近0）
print("均值：", np.mean(random_data))      # -0.053469588354015586
# 输出≈1（随机，接近1）
print("标准差：", np.std(random_data))     # 1.0180882830808482
