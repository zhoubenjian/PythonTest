'''
分类问题（随机森林分类） - 鸢尾花分类
    优点：
        1.可解释性强，可以画出树，看懂每一步判断逻辑
        2.**不需要特征归一化、标准化**，不受量纲影响
        3.能捕捉非线性关系；可处理连续、离散特征
        4.能输出特征重要性

    缺点：
        1.极易过拟合，对训练集微小变化敏感
        2.容易偏向取值很多的特征
        3.不擅长高维噪声数据
        4.容易生成偏向样本多的类别，不平衡数据集要注意
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
