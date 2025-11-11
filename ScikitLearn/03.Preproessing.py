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



