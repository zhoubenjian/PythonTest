'''
归一化：通常指将数据线性地缩放到一个固定的区间，最常见的是 [0, 1]。
常用方法：Min-Max 归一化
公式：
X_normalized = (X - X_min) / (X_max - X_min)

X_min：该特征列的最小值

X_max：该特征列的最大值

对于新数据点，如果其值超出原始最小最大值范围，归一化后可能会超出[0, 1]。



标准化：通常指将数据转换为均值为0，标准差为1的标准正态分布（或Z-score分布）。
常用方法：Z-Score 标准化
公式：
X_standardized = (X - μ) / σ

μ：该特征列的均值

σ：该特征列的标准差
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data = np.array([[
    2.1, 9.9, 8.1,
    1.9, 1.1, 4.9,
    1.7, 3.9, 4.1
]])



# 归一化
# 1.实例化转换器类
transfer1 = MinMaxScaler(feature_range=(0, 1))
# 2.进行转换
transfer_data1 = transfer1.fit_transform(data)
print(f'归一化值：{transfer_data1}')

# 标准化
# 1.实例化转换器类
transfer2 = StandardScaler()
# 2.进行转换
transfer_data2 = transfer2.fit_transform(data)
print('标准化值：%s' % (transfer_data2))
print('均值：%s' % (transfer2.mean_))
print('标准差：%s' % (transfer2.scale_))



