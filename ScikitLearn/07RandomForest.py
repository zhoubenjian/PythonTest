'''
随机森林
    核心思想“三个臭皮匠，顶个诸葛亮”

    优点：
        适用于分类和回归
        能处理非线性关系
        对特征缩放不敏感
        通常具有较好的泛化能力
        可评估特征重要性

    缺点：
        模型体积可能较大
        预测速度比单棵决策树慢
        可解释性较弱
        高维稀疏数据上未必优于线性模型
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


'''
1.加载数据
'''
iris = load_iris()
# 特征：花萼长度，花萼宽度，花瓣长度，花瓣宽度
X = iris.data
# 标签：山鸢尾0（setosa）、变色鸢尾1（versicolor）、维吉尼亚鸢尾2（virginica）
y = iris.target


'''
2.数据划分
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''
3.创建随机森林分类器
'''
# n_estimators(决策树数量，默认100)，max_depth(单棵决策树的最大深度，防止过拟合)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)


'''
4.训练模型
'''
rf_clf.fit(X_train, y_train)


'''
5.测试集验证
'''
y_pred = rf_clf.predict(X_test)


'''
6.模型评估
'''
print('测试集准确率：', accuracy_score(y_test, y_pred), sep='')
print('分类报告：\n', classification_report(y_test, y_pred, target_names=iris.target_names), sep='')