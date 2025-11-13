'''
KNN实现鸢尾花识别（交叉验证）
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier



# 1.获取数据
iris = load_iris()

# 2.数据基本处理
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)

# 3.特征工程 - 特征预处理
transfer = StandardScaler()
x_train_transfer = transfer.fit(x_train)
x_test_transfer = transfer.fit(x_test)

# 4.机器学习KNN
# 4.1实例化一个估计器
estimator = KNeighborsClassifier()
# 4.1模型调优-交叉验证，网格搜索
param_grid = {'n_neighbors': [2, 3, 5, 7, 9]}
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=5)
# 4.3模型训练
estimator.fit(x_train, y_train)

# 5.模型评估
# 5.1 预测值输出结果
y_pred = estimator.predict(x_test)
print('预测值:\n', y_pred)
print('预测值和真实值的的对比:\n', y_pred == y_test)
# 5.2 准确率计算
score = estimator.score(x_test, y_test)
print(f'准确率为:{score:.2f}')
# 5.3 查看交叉验证，网络搜索一些属性
print(f'交叉验证中，得到最好的结果：{estimator.best_estimator_}')
print(f'交叉验证中，得到最好的模型：{estimator.best_params_}')
print(f'交叉验证中，得到的模型结果是：{estimator.cv_results_}')