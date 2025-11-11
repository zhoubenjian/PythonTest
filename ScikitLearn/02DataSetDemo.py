from sklearn.datasets import load_iris, fetch_20newsgroups
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



# 1.数据集获取
# 1.1小数据集获取
iris = load_iris()

# 1.2大数据集合获取
# news = iris.feature_names
# print(news)


# 2.数据集属性描述
print("数据集特征值:\n", iris.data)
print("数据集目标值:\n", iris["target"])      # iris["target"] <=> iris.target
print("数据集特征值:\n", iris.feature_names)
print("数据集目标值:\n", iris.target_names)
# print("数据集描述:\n", iris.DESCR)


# 加载鸢尾花数据
# 3.数据可视化
# 数据转为dataframe格式
iris_d = pd.DataFrame(iris.data, columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
iris_d["target"] = iris.target
# print(iris_d)
# iris_d['Species'] = iris.target

def iris_plot(iris, col1, col2):
    sns.lmplot(x=col1, y=col2, data=iris, hue='target', fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("鸢尾花种类分布图")
    plt.grid(False)  # 关闭网格线
    plt.show()

# iris_plot(iris_d, 'Petal Width', 'Sepal Length')


# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
print('训练集特征值:\n', x_train)
print('训练集目标值:\n', y_train)
print('测试机特征值:\n', y_test)
print('测试机目标值:\n', y_test)

print('测试集形状:\n', y_train.shape)
print('训练集形状:\n', y_test.shape)