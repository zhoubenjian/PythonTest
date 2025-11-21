'''
 分类问题（随机森林分类） - 鸢尾花分类
'''
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
rac = RandomForestClassifier(n_estimators=10, random_state=42)
# 训练模型
rac.fit(X_train, y_train)

# 预测
y_pred = rac.predict(X_test)

# 评估
print(f'准确率：{accuracy_score(y_test, y_pred):.2f}')
print('分类报告：')
print(classification_report(y_test, y_pred))
