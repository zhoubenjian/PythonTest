import numpy as np


class Perceptron:
    def __init__(self, lr=0.1, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(xi, self.w) + self.b) <= 0:
                    # 更新权重向量w
                    self.w += self.lr * yi * xi
                    # 更新偏置b
                    self.b += self.lr * yi
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)


if __name__ == "__main__":
    # 示例数据
    X = np.array([
        [2, 3],
        [1, 1],
        [4, 5],
        [-1, -2],
        [-3, -4],
        [-2, -1]
    ])

    y = np.array([1, 1, 1, -1, -1, -1])

    model = Perceptron()
    model.fit(X, y)

    print(f'权重：{model.w}')
    print(f'偏置：{model.b}')
    print(f'预测：{model.predict(X)}')
