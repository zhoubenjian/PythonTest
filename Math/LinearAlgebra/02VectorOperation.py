'''
向量（矩阵）运算
    +, -：必须是同型矩阵
'''
import numpy as np


a = np.array([2, 3])
b = np.array([5, 7])

# 向量加法
print(f'a + b = {a + b}')       # [7 10]

# 向量减法（等价于加法的负向量）
print(f'a - b = {a - b}')       # [-3 -4]

# 向量数乘
print(f'2.5 * a = {2.5 * a}')   # [5.  7.5]

# 相反（负）向量
print(f'-a = {-a}')             # [-2, -3]

# 交换率 a⋅b=b⋅a
print(f'{np.allclose(a + b, b + a)}')       # True

# 分配率 a⋅(b+c)=a⋅b+a⋅c
c = np.array([9, 11])
print(f'{np.allclose((a + b) @ c, a @ c + b @ c)}')       # True

# 自身点积 a⋅a = |a|^2
print(f'{np.allclose(a @ a, np.linalg.norm(a) ** 2)}')      # True


print('\n' + '-' * 20 + '\n')


'''
向量点乘（内积）
'''
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 方法1：np.dot()
print(np.dot(a, b))     # 32

# 方法2：@（推荐）
print(a @ b)     # 32

# 方法3：手动实现
print(sum(a[i] * b[i] for i in range(len(a))))      # 32


'''
几何定义反推夹角
'''
# a向量模长 |a|
norm_a = np.linalg.norm(a)
# b向量模长 |b|
norm_b = np.linalg.norm(b)
print(f'|a| = {norm_a:.2f}, |b| = {norm_b:.2f}')

# 向量a，b夹角的余弦值 cos(θ) = a·b / |a|*|b|
cos_theta = np.dot(a, b) / (norm_a * norm_b)
print(f'a,b夹角的余弦值 cos(θ) = {cos_theta:.4f}')

'''
2π rad = 360° <=> 1 rad = 180° / π <=> 1° = π / 180 rad
'''
# 夹角（角度）
angle_theta = np.degrees(np.arccos(cos_theta))
# 夹角（弧度）
radian_theta = np.radians(np.arccos(cos_theta))
print(f'向量a，b夹角θ（角度）：{angle_theta:.1f}°\n向量a，b夹角θ（弧度）：{radian_theta:.4f}rad')       # 12.9°     0.0039rad


print('\n' + '*' * 50 + '\n')


'''
几何验证
'''
verify = norm_a * norm_b * cos_theta
print(f'几何验证(||a||·||b||·cosθ == a @ b): {np.allclose(verify, a @ b)}')     # True


print('\n' + '-' * 30 + '\n')


'''
不同夹角
'''
# 锐角（θ < 90°） 正值
print(f'[1, 1] * [2, 2] = {np.dot([1, 1], [2, 2])}')     # 4
# 直角（θ = 90°） 0
print(f'[1, 0] * [0, 1] = {np.dot([1, 0], [0, 1])}')     # 0
# 锥角（θ > 90°） 负值
print(f'[1, -1] * [-2, 2] = {np.dot([1, 1], [-2, -2])}') # -4



