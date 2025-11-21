from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



# 加载数据
iris = load_iris()
X = iris.data       # 特征
y = iris.target     # 目标值

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 实例化决策时（最大深度3）
clf = DecisionTreeClassifier(max_depth=3)

# 模型训练
clf = clf.fit(X_train, y_train)

# 模型预测
y_pred = clf.predict(X_test)

# 模型评估
print(f'准确率: {accuracy_score(y_test, y_pred):.2f}')
print('分析报告：')
print(classification_report(y_test, y_pred))


# 可视化
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title('鸢尾花分类决策树（Decision Tree）')
plt.show()