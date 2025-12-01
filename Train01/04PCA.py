'''
PCA（主要成分分析）分析葡萄酒主要成分
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


print("=" * 60)
print("使用sklearn的PCA进行葡萄酒数据降维")
print("=" * 60)


# 1. 加载数据
print("\n1. 加载数据...")
wine = load_wine()
X = wine.data       # 特征值
y = wine.target     # 目标值（标签）
feature_names = wine.feature_names      # 特征名称
target_names = wine.target_names        # 目标名称

print(f'数据形状:{X.shape}')
print(f'特征数:{len(feature_names)}')
print(f'目标数:{len(target_names)}')
print(f'特征名称:{feature_names}')


# 2. 数据标准化
print("\n2. 数据标准化...")
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print('标准化完成！')


# 3. 使用sklearn进行PCA分析
print('\n3. 进行PCA分析...')

# 首先分析所有主成分，查看方差解释情况
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_std)

# 方差解释率（默认从大到小排列）
explained_variance_ratio = pca_full.explained_variance_ratio_
# 计算累计方差解释率
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("各主成分方差解释率：")
for i, (var_radio, cum_radio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
    print(f'PC{i + 1:2d}: {var_radio:.4f}({var_radio * 100:6.2f}%) | 累计: {cum_radio:.4f}({cum_radio * 100:6.2f}%)')


# 4. 可视化方差解释率
print("\n4. 生成可视化图表...")

plt.figure(figsize=(15, 10))

# 4.1 碎石图
plt.subplot(2, 2, 1)
components = range(1, len(explained_variance_ratio) + 1)
plt.bar(components, explained_variance_ratio, alpha=0.7, color='lightblue', label='单个贡献率')
plt.plot(components, explained_variance_ratio, 'ro-', linewidth=2, markersize=6, label='趋势线')
plt.xlabel('主成分')
plt.ylabel('方差解释率')
plt.title('主成分分析 - 碎石图')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.2 累计方差解释率
plt.subplot(2, 2, 2)
plt.plot(components, cumulative_variance_ratio, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% 阈值')
plt.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='85% 阈值')
plt.axhline(y=0.70, color='green', linestyle='--', alpha=0.7, label='70% 阈值')
plt.xlabel('主成分数量')
plt.ylabel('累计方差解释率')
plt.title('累计方差解释率')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. 选择前2个主成分进行降维
print("\n5. 使用前2个主成分进行降维...")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_std)

print(f"降维后数据形状: {X_pca_2d.shape}")
print(f"PC1解释方差: {pca_2d.explained_variance_ratio_[0]:.4f} ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)")
print(f"PC2解释方差: {pca_2d.explained_variance_ratio_[1]:.4f} ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)")
print(f"总解释方差: {sum(pca_2d.explained_variance_ratio_):.4f} ({sum(pca_2d.explained_variance_ratio_)*100:.2f}%)")


# 6. 可视化二维降维结果
plt.subplot(2, 2, 3)
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']

for i, (color, marker, target_name) in enumerate(zip(colors, markers, target_names)):
    plt.scatter(X_pca_2d[y == i, 0],
                X_pca_2d[y == i, 1],
                c=color,
                marker=marker,
                label=target_name,
                alpha=0.7,
                edgecolors='white',
                s=80)

plt.xlabel(f'第一主成分 (PC1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'第二主成分 (PC2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('葡萄酒数据 - PCA二维投影')
plt.legend()
plt.grid(True, alpha=0.3)


# 7. 分析主成分载荷
print("\n6. 分析主成分载荷...")
loadings = pca_2d.components_.T

print("\n主成分载荷矩阵 (特征向量):")
print("特征名称".ljust(20) + "PC1载荷".ljust(12) + "PC2载荷".ljust(12) + "PC1贡献".ljust(12) + "PC2贡献")
print("-" * 70)

for i, (feature, loading) in enumerate(zip(feature_names, loadings)):
    pc1_contribution = abs(loading[0])
    pc2_contribution = abs(loading[1])
    print(f"{feature:20} {loading[0]:11.3f} {loading[1]:11.3f} {pc1_contribution:11.3f} {pc2_contribution:11.3f}")


# 8. 可视化特征载荷
plt.subplot(2, 2, 4)
# 绘制散点图背景
for i, (color, marker, target_name) in enumerate(zip(colors, markers, target_names)):
    plt.scatter(X_pca_2d[y == i, 0],
                X_pca_2d[y == i, 1],
                c=color,
                marker=marker,
                label=target_name,
                alpha=0.2,
                s=40)

# 添加特征向量箭头
for i, (feature, loading) in enumerate(zip(feature_names, loadings)):
    plt.arrow(0, 0, loading[0]*7, loading[1]*7,
              color='black', alpha=0.8, head_width=0.2, linewidth=1.5)
    plt.text(loading[0]*7.5, loading[1]*7.5,
             feature, fontsize=8, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

plt.xlabel(f'第一主成分 (PC1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'第二主成分 (PC2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA投影与特征载荷向量')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 9. 详细的主成分解释
print("\n7. 主成分解释分析...")

# 找出对PC1贡献最大的特征
print("\n对第一主成分(PC1)贡献最大的特征:")
pc1_contributions = sorted(zip(feature_names, loadings[:, 0], abs(loadings[:, 0])),
                          key=lambda x: x[2], reverse=True)
for feature, loading, abs_loading in pc1_contributions[:5]:
    direction = "正向" if loading > 0 else "负向"
    print(f"  {feature:15s}: {loading:7.3f} ({direction})")

# 找出对PC2贡献最大的特征
print("\n对第二主成分(PC2)贡献最大的特征:")
pc2_contributions = sorted(zip(feature_names, loadings[:, 1], abs(loadings[:, 1])),
                          key=lambda x: x[2], reverse=True)
for feature, loading, abs_loading in pc2_contributions[:5]:
    direction = "正向" if loading > 0 else "负向"
    print(f"  {feature:15s}: {loading:7.3f} ({direction})")


# 10. 三维PCA可视化（可选）
print("\n8. 生成三维PCA可视化...")

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_std)

fig = plt.figure(figsize=(12, 5))

# 三维散点图
ax1 = fig.add_subplot(121, projection='3d')
for i, (color, marker, target_name) in enumerate(zip(colors, markers, target_names)):
    ax1.scatter(X_pca_3d[y == i, 0],
                X_pca_3d[y == i, 1],
                X_pca_3d[y == i, 2],
                c=color,
                marker=marker,
                label=target_name,
                alpha=0.7,
                s=60)

ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax1.set_title('三维PCA投影')
ax1.legend()

# 不同视角的三维图
ax2 = fig.add_subplot(122, projection='3d')
for i, (color, marker, target_name) in enumerate(zip(colors, markers, target_names)):
    ax2.scatter(X_pca_3d[y == i, 0],
                X_pca_3d[y == i, 1],
                X_pca_3d[y == i, 2],
                c=color,
                marker=marker,
                label=target_name,
                alpha=0.7,
                s=60)

ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax2.set_title('三维PCA投影（不同视角）')
ax2.view_init(30, 45)  # 改变视角
ax2.legend()

plt.tight_layout()
plt.show()

print(f"三维PCA累计解释方差: {sum(pca_3d.explained_variance_ratio_):.4f} ({sum(pca_3d.explained_variance_ratio_)*100:.2f}%)")

# 11. 最终总结
print("\n" + "=" * 60)
print("PCA降维分析总结")
print("=" * 60)
print(f"✓ 原始数据维度: {X.shape[1]}个特征")
print(f"✓ 降维后维度: 2个主成分")
print(f"✓ 保留的方差信息: {sum(pca_2d.explained_variance_ratio_)*100:.2f}%")
print(f"✓ 数据压缩率: {(1 - 2/X.shape[1])*100:.2f}%")
print(f"✓ 三个葡萄酒类别在二维空间中清晰可分")
print("\n主要结论:")
print("- PCA成功将13维数据降至2维，同时保留了大部分重要信息")
print("- 前两个主成分能够有效区分不同品种的葡萄酒")
print("- 可视化结果显示明显的聚类效果，证明PCA的有效性")
print("- 该方法可用于后续的分类任务或探索性数据分析")
