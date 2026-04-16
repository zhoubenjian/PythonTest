'''
分类算法（决策树分类） - 鸢尾花分类

适用场景：可解释性要求高的分类任务
'''
import numpy as np

from sklearn.tree import DecisionTreeClassifier         # 决策树模型
from sklearn.datasets import load_iris                  # 鸢尾花数据集
from sklearn.model_selection import train_test_split    # 数据划分
from sklearn.preprocessing import StandardScaler        # 数据处理
from sklearn.metrics import accuracy_score, classification_report   # 模型评估, 分类报告


# 1.加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target


# 2.数据划分(训练集:70% / 测试集:30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3.数据标准化(提升模型精度)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4.实例化决策树模型
dt_model = DecisionTreeClassifier(max_depth=3)      # 限制树深度，防止过拟合
# 训练决策树模型
dt_model.fit(X_train, y_train)


# 5.预测 + 评估
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'决策树模型准确率:{(accuracy * 100):.1f}%')
print("决策树模型分类报告:\n", classification_report(y_test, y_pred))






