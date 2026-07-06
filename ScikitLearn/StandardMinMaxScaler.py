'''
标准化（Z-Score Normalization）:把数据调整为均值为0，标准差为1的标准正态分布
    算法依赖距离（如KNN、K-Means、SVM），且数据本身边界明确（如图像像素值 0-255）。
    需要使用梯度下降且特征范围差异极大（归一化能让收敛更快）。
    需要保证输出在特定区间（如某些激活函数的输入）


归一化（Min-Max Scaling）:把数据缩放到一个固定的范围，通常是 [0, 1] 或 [-1, 1]
    数据服从或近似正态分布。
    算法假设数据呈正态分布（如线性回归、逻辑回归、LDA）。
    降维（PCA）前必须标准化，否则方差大的特征会主导主成分。
    数据存在明显异常值（标准化能降低其影响）。
'''


from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = [[10, 19, 51, 99], [11, 91, 31, 17]]

# 标准化
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print(f'标准化后：\n{X_std}')


print('-' * 20)


# 归一化
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)
print(f'归一化后：\n{X_minmax}')




