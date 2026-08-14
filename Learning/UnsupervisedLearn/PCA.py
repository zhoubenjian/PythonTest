'''
无监督学习：
    PCA（主要成分分析）：
    PCA 是一种降维技术，它通过线性变换将数据转换到新的坐标系中，使得大部分的方差集中在前几个主成分上。
    应用场景：图像降维、特征选择、数据可视化。
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

# 3. 查看信息保留情况
print(f"各主成分方差比例: {pca.explained_variance_ratio_}")            # 各主成分方差比例: [0.92461872 0.05306648]
print(f"累计方差比例: {sum(pca.explained_variance_ratio_):.2%}")      # 累计方差比例: 97.77%


# 4. 可视化（带图例）
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1],
                c=color, label=iris.target_names[i])
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA降维可视化（4维 => 2维）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()