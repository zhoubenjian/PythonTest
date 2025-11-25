'''
对一个二维或高维数据集进行PCA（主要充分分析），将其降到 k 维（比如 2D → 1D 或 100D → 2D），并可视化结果。
'''
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def pca(X, k):
    """
        使用特征分解实现 PCA

        参数:
            X: shape (n_samples, n_features) 的数据矩阵
            k: 降维后的维度 (k <= n_features)

        返回:
            X_pca: 降维后的数据，shape (n_samples, k)
            components: 主成分（特征向量），shape (k, n_features)
            explained_variance: 每个主成分解释的方差（特征值）
    """
    # 1. 数据标准化：每列（特征）减去均值
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # 2. 计算协方差矩阵 (n_features x n_features)
    n_samples = X.shape[0]
    cov_matrix = np.cov(X_centered, rowvar=False)   # rowvar=False 表示每行是一个样本

    # 3. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # 注意：eigh 用于对称矩阵，返回按升序排列的特征值

    # 4. 按特征值从大到小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. 取前 k 个主成分
    components = eigenvectors[:, :k].T  # shape (k, n_features)

    # 6. 投影到主成分空间
    X_pca = X_centered @ components.T  # shape (n_samples, k)

    return X_pca, components, eigenvalues[:k]


# -------------------------
# 示例：2D 数据降维到 1D
# -------------------------
if __name__ == "__main__":
    # 生成模拟数据：椭圆分布
    np.random.seed(42)
    mean = [0, 0]
    cov = [[3, 2], [2, 2]]  # 协方差矩阵，有相关性
    X = np.random.multivariate_normal(mean, cov, 200)  # shape (200, 2)

    # 执行 PCA，降到 1 维
    X_pca, components, var = pca(X, k=1)

    print("原始数据形状:", X.shape)
    print("降维后形状:", X_pca.shape)
    print("第一主成分方向:", components[0])
    print("解释方差:", var[0])

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='原始数据')

    # 画出第一主成分方向（通过均值点）
    mean_point = np.mean(X, axis=0)
    direction = components[0] * 3  # 放大便于显示
    plt.arrow(mean_point[0], mean_point[1],
              direction[0], direction[1],
              head_width=0.3, head_length=0.4,
              fc='red', ec='red', linewidth=2, label='第一主成分')

    # 将降维后的点映射回原始空间（用于可视化投影）
    X_recon = X_pca @ components + mean_point
    plt.scatter(X_recon[:, 0], X_recon[:, 1], c='orange', s=10, label='PCA 投影点')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('PCA 降维可视化（2D → 1D）')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()