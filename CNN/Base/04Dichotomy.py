'''
感知机（二分法示例）
'''
import numpy as np


# 感知机
class Perceptron:
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    # 激活函数
    def activation(self, z):
        return np.where(z > 0, 1, -1)

    # 训练方法
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        for _ in range(self.epochs):
            errors = 0
            for i, x_i in enumerate(X):
                z = np.dot(x_i, self.w) + self.b
                y_pred = self.activation(z)

                # 分类错误
                if y[i] * y_pred <= 0:
                    # 参数更新（核心！）
                    self.w += self.lr * y[i] * x_i      # 更新权重w
                    self.b += self.lr * y[i]            # 更新偏置b
                    errors += 1

            # 如果没有分类错误，提前结束训练
            if errors == 0:
                break


    # 预测函数
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return self.activation(z)



# 测试：线性可分数据（AND逻辑）
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
# AND标签
y = np.array([-1, -1, -1, 1])
model = Perceptron(lr=0.1, epochs=100)
model.fit(X, y)
print("权重:", model.w, "偏置:", model.b)
print("预测:", model.predict(X))