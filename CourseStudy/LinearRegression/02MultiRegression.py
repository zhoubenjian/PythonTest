'''
多因子线性回归
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 加载数据
data = pd.read_csv('./data/usa_housing_price.csv')
print(data.head())      # head默认预览前五行

# 可视化展示
plt.figure(figsize=(10, 10))

# 面积 价格可视化
r3_squareFeet = plt.subplot(423)
plt.scatter(data.loc[:, 'SquareFeet'], data.loc[:, 'Price'])
plt.title('SquareFeet vs Price')

# 卧室数量 价格可视化
r4_bedrooms = plt.subplot(424)
plt.scatter(data.loc[:, 'Bedrooms'], data.loc[:, 'Price'])
plt.title('Bedrooms vs Price')

# 洗手间数量 价格可视化
r5_bathrooms = plt.subplot(425)
plt.scatter(data.loc[:, 'Bathrooms'], data.loc[:, 'Price'])
plt.title('Bathrooms vs Price')

# 建造年份 价格可视化
r6_yearBuilt = plt.subplot(426)
plt.scatter(data.loc[:, 'YearBuilt'], data.loc[:, 'Price'])
plt.title('YearBuilt vs Price')

plt.show()


# 多因子拟合
X_multi = data.drop(['Price'], axis=1)
y = np.array(data.loc[:, 'Price'])
print(X_multi.shape, y.shape)

lr = LinearRegression()
lr.fit(X_multi, y)
y_pred = lr.predict(X_multi)
print(mean_squared_error(y, lr.predict(X_multi)))

r7_mulity = plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred)
plt.show()





