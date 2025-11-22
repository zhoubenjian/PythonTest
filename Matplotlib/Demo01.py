import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False



X = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.figure(figsize=(12, 8))
# 直线图
plt.plot(X, y, color='red')
# 散点图
plt.scatter(X, y, color='blue', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = 2 * x')
plt.show()
