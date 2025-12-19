'''
SVM（支持向量机）
'''
# 数据集模块
from sklearn import datasets
# 模型选择模块，用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 预处理模块，用于数据标准化
from sklearn.preprocessing import StandardScaler
# 导入模型（这里用支持向量机 SVM）
from sklearn.svm import SVC
# 导入评估指标
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


'''
步骤 2: 加载数据 
'''
# 加载鸢尾花数据集
iris = datasets.load_iris()
# 特征数据 (150个样本，4个特征)
X = iris.data
# 标签数据 (3种类别：Setosa, Versicolour, Virginica)
y = iris.target
# 查看特征名称和标签名称
print("特征名：", iris.feature_names)
print("标签名：", iris.target_names)


'''
步骤 3: 划分训练集和测试集
'''
# 随机将 30% 的数据划分为测试集，设置 random_state 保证每次划分结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


'''
步骤 4: 数据预处理
'''
# 实例化一个标准化器
scaler = StandardScaler()
# 拟合训练数据并计算均值和方差
scaler.fit(X_train)
# 用计算好的均值和方差来转换训练数据和测试数据
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
# 注意：一定要用训练集的参数来转换测试集，避免数据泄露


'''
步骤 5: 创建、训练（拟合）模型
'''
# 实例化一个支持向量机分类器，设置核函数和正则化参数
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# 在标准化后的训练数据上拟合模型
svm_model.fit(X_train_std, y_train)


'''
步骤 6: 进行预测
'''
# 对测试集进行预测
y_pred = svm_model.predict(X_test_std)


'''
步骤 7: 评估模型性能
'''
# 1. 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率： {accuracy:.4f}") # 输出：模型准确率： 0.9778

# 2. 查看混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵：\n", conf_matrix)

# 3. 生成详细的分类报告
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("分类报告：\n", class_report)