'''
K临近算法
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1.加载数据 并预处理
iris = load_iris()      # 鸢尾花数据数据集（150个样本，4个特征，3个类别）
X = iris.data           # 特征数据（花萼长度，花萼宽度，花瓣长度，花瓣宽度）
y = iris.target         # 标签（映射类别名称0=setosa，1=versicolor，2=virginica）

# 划分训练集和测试集（训练集70%，测试集30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # random_state固定随机种子，保证结果可复现


# 2.创建并训练KNN模型
knn = KNeighborsClassifier(n_neighbors = 5)   # 初始化KNN分类器，设置K值为3（可根据需求调整）
knn.fit(X_train, y_train)               # 用训练集训练模型（KNN是惰性学习，这里实际只是存储训练数据）


# 3.预测结果与评估
y_pred = knn.predict(X_test)                  # 测试集预测

# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN（模型）的预测准确性：{accuracy:.2f}')


# 4.单个新样本预测示例
# 构造一个新样本（特征：花萼长5.1，花萼宽3.5，花瓣长1.4，花瓣宽0.2）
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# 预测类别
pred_label = knn.predict(new_sample)

# 映射类别名称（0=setosa，1=versicolor，2=virginica）
label_name = iris.target_names[pred_label][0]
print(f"新样本的预测类别: {pred_label[0]} ({label_name})")






