'''
无监督学习：
    PCA（主要成分分析）
'''
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


# 1.加载鸢尾花数据集
iris = load_iris()
X = iris.data       # 特征矩阵：150个样本 × 4个特征（花萼长宽、花瓣长宽）
y = iris.target     # 标签：3种鸢尾花（0,1,2）


# 2.降维到二维
pca = PCA(n_components = 2)     # 关键参数：指定降维后的维度为2
X_pca = pca.fit_transform(X)    # 执行降维：150 × 4 => 150 × 2

# 查看各主成分解释的方差比例
print(pca.explained_variance_ratio_)    # [0.92461872 0.05306648]       第1主成分保留92.46%信息，第2主成分保留5.31%

# 查看主成分方向（特征向量）
'''
[[ 0.36138659 -0.08452251  0.85667061  0.3582892 ]
 [ 0.65658877  0.73016143 -0.17337266 -0.07548102]]
'''
print(pca.components_)          # 2 × 4矩阵，每一行是一个主成分


# 3.可视化结果
'''
X_pca[:, 0]：第1主成分（X轴）
X_pca[:, 1]：第2主成分（Y轴）
c=y：用不同颜色标记3种鸢尾花
'''
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = y, cmap='viridis')
plt.title('PCA of Iris Dataset')
plt.show()