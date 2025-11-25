'''
协方差
'''
import numpy as np


# 假设我们有一个数据集，有3个特征（身高、体重、年龄）和5个样本
# 数据格式为 (samples, features) -> (5, 3)
data = np.array([
    [170, 65, 25],  # 样本1
    [180, 75, 30],  # 样本2
    [165, 55, 22],  # 样本3
    [175, 70, 27],  # 样本4
    [185, 80, 35]   # 样本5
])

'''
如果你的数据是 (n_features, n_samples)，使用默认的 rowvar=True
如果你的数据是 (n_samples, n_features)，务必设置 rowvar=False
'''
# 因为我们的数据是 (samples, features)，所以需要设置 rowvar=False
cov_matrix = np.cov(data, rowvar=False)

print("数据形状：", data.shape)
print('协方差形状：%s' % str(cov_matrix.shape))
print('\n协方差矩阵：')
print(cov_matrix)

# 身高的方差（第一个特征）
height_variance = cov_matrix[0, 0]
print(f'\n身高方差：{height_variance:.2f}')

# 身高和体重的协方差
height_weight_cov = cov_matrix[0, 1]
print(f'身高和体重的协方差：{height_weight_cov:.2f}')

# 体重和年龄的协方差
weight_age_cov = cov_matrix[1, 2]
print(f'体重和年龄的协方差：{weight_age_cov:.2f}')
