'''
scipy.optimize.curve_fit：拟合数据
scipy.optimize.minimize / scipy.optimize.root：求函数最小值/根
scipy.integrate.quad：计算积分
scipy.stats：做概率计算/统计检验
scipy.interpolate：补全缺失的数据点
scipy.sparse：处理大型稀疏矩阵
'''
from scipy.optimize import minimize
from scipy.integrate import quad    # 求导
from scipy import linalg            # 矩阵计算
import numpy as np


'''
函数最小值
'''
# 定义函数
def f1(x):
    return (x - 3) ** 2

# 猜测从x=0开始寻找最小值
min_result = minimize(f1, 0)

# 输出
print(f'f(x) = (x-3)^2最小值：{min_result.x}')     # [2.99999998]


print('-' * 40)


'''
积分
'''
#
def f2(x):
    return np.sin(x);

'''
函数：sin(x)
积分下限：0
积分上限：π
'''
quad_result, error_estimate = quad(f2, 0, np.pi)

print(f'积分结果：{quad_result:.2f}')
print(f'估计误差：{error_estimate}')


print('-' * 40)


'''
线性代数
'''
A = np.array([
    [3, 1],
    [1, 2]
])

print(f'A行列式值：{linalg.det(A)}')

eigen_values, eigen_vectors = linalg.eig(A)
print(f'A特征值：{eigen_values}')
print(f'A特征向量：{eigen_vectors}')


print('-' * 40)


'''
解线性方程组
3 * x + y = 9
x + 2 * y = 8
'''
# 系数矩阵A
A = np.array([
    [3, 1],
    [1, 2]
])

# 右侧常数b
b = np.array([9, 8])

# 求解
x = linalg.solve(A, b)
print(f'解：x = {x[0]:.1f}, y={x[1]:.1f}')    # 解：x = 2.0, y=3.0