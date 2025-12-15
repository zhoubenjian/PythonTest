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
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)


# 示例数据（线性可分）
X = np.array([[2, 3], [1, 1], [3, 4], [-1, -2], [-2, -1], [-3, -3]])
y = np.array([1, 1, 1, -1, -1, -1])

model = Perceptron()
model.fit(X, y)
print("权重:", model.w)
print("偏置:", model.b)
print("预测:", model.predict(X))

