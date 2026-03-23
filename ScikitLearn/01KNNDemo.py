'''
KNN(K邻近算法)
'''
# 导入模块
from sklearn.neighbors import KNeighborsClassifier


# 1.构造数据集
x = [[0], [3], [9], [12]]
y = [0, 30, 90, 120]


# 2.训练模型
# 2.1 实例化一个估计对象
estimator = KNeighborsClassifier(n_neighbors=1)

# 2.2 调用fit方法，进行训练
estimator.fit(x, y)


# 3.数据预测
res1 = estimator.predict([[4]])
print(res1)  # [30]

res2 = estimator.predict([[8]])
print(res2)  # [90]

res3 = estimator.predict([[14]])
print(res3)  # [120]