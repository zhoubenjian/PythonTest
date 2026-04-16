'''
二分类（逻辑回归）
做分类任务，预测离散类别（本质是线性回归 + sigmoid）

任务类型：分类（二分类为主）
输出结果：0~1 之间的概率
核心公式：y=σ(wx+b)
激活函数：Sigmoid——max(0, wx+b)
损失函数：交叉熵损失 Cross Entropy
假设分布：伯努利分布
典型场景：垃圾邮件识别、患病判断、点击率预测
'''
import numpy as np
from sklearn.linear_model import LogisticRegression


# 1.模拟数据
# X: 输入特征
X = np.array([
    [1], [2], [3], [4], [5]
])

# y: 目标变量   0/1 类别
y = np.array([0, 0, 0, 1, 1])


# 实例化模型
model = LogisticRegression()
# 训练模型
model.fit(X, y)


# 预测类别[[分类类型]]
print("预测类别:", model.predict([[3.5]]))          # 预测类别: [0]

# 预测概率 [[0的概率, 1的概率]]
# predict_proba 输出顺序 永远 = 你训练时 y 里的类别顺序
print("预测概率:", model.predict_proba([[3.5]]))    # 预测概率: [[0.5208689 0.4791311]]
