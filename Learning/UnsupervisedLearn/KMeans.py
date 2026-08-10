'''
K-means 聚类（K-means Clustering）:
    K-means 是一种基于中心点的聚类算法，通过不断调整簇的中心点，使每个簇中的数据点尽可能靠近簇中心。

    应用场景: 客户分群、市场分析、图像压缩。
'''

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from  sklearn.datasets import make_blobs


# 1.生成简单二维数据集
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


# 2.训练K-meas模型
model = KMeans(n_clusters=4, random_state=0)
model.fit(X)


# 3.预测聚类结果
y_kmeas = model.predict(X)


# 4.可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c = y_kmeas, s=50, cmap='viridis')
plt.show()