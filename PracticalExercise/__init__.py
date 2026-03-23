import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1.加载数据 并预处理
iris = load_iris()  # 150个样本 4个特征 3个类别
X = iris.data       # 特征（花萼长度 花萼宽度 花瓣长度 花瓣宽度）
y = iris.target     # 标签（映射类别名称0=setosa，1=versicolor，2=virginica）

# 划分数据集（训练集：70%）
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)


# 2.创建knn对象 并训练
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

# 计算预测结果 准确率
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN（模型）准确率：{accuracy:.2f}')


# 3.校验
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # 构造一个新样本（特征：花萼长5.1，花萼宽3.5，花瓣长1.4，花瓣宽0.2）

# 预测类别
pred_label = knn.predict(new_sample)

# 映射类别名称（0=setosa，1=versicolor，2=virginica）
label_name = iris.target_names[pred_label][0]
print(f'新样本的预测类别：{pred_label[0]}({label_name})')

