'''
KNN(K邻近算法)
'''
# 导入模块
from sklearn.neighbors import KNeighborsClassifier


# 1.构造数据集
x = [[0], [1], [2], [3]]
y = [0, 10, 20, 30]


# 2.训练模型
# 2.1 实例化一个估计对象
estimator = KNeighborsClassifier(n_neighbors=1)

# 2.2 调用fit方法，进行训练
estimator.fit(x, y)


# 3.数据预测
res1 = estimator.predict([[2]])
print(res1)  # [20]

res2 = estimator.predict([[5]])
print(res2)  # [30]