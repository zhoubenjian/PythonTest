'''
感知机（MLP）
'''
import numpy as np


class Perceptron:
    def __init__(self, learn_rate = 0.01, max_iter = 1000):
        self.lr = learn_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(xi, self.w) + self.b) <= 0:
                    # 参数更新（核心！）
                    self.w += self.lr * yi * xi     # 更新权重
                    self.b += self.lr * yi          # 更新偏置
                    errors += 1
            if errors == 0:
                break

    # 预测
    def predict(self, X):
        # NumPy 库中用于计算数组元素符号值的函数
        return np.sign(np.dot(X, self.w) + self.b)  # 正数返回1 负数返回-1 0返回0（包括 0.0、-0.0）


# 示例数据（线性可分）
X = np.array([[2, 3], [1, 1], [3, 4], [-1, -2], [-2, -1], [-3, -3]])
y = np.array([1, 1, 1, -1, -1, -1])

model = Perceptron()
model.fit(X, y)
print("权重:", model.w)
print("偏置:", model.b)
print("预测:", model.predict(X))

