import numpy as np


class Perceptron:

    # 初始化
    def __init__(self, input_size, learn_rate = 0.1, max_iter = 100):
        self.w = np.zeros(input_size)
        self.b = 0.0
        self.lr = learn_rate
        self.max_iter = max_iter

    # 激活函数
    def sign(self, x):
        return 1 if x >= 0 else -1

    # 训练（学习规则核心）
    def fit(self, X, y):
        for _ in range(self.max_iter):
            updated = False
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]

                # 预测
                y_pred = self.sign(np.dot(self.w, x_i) + self.b)

                # 分类错误 更新规则
                if y_i * y_pred <= 0:
                    self.w += self.lr * y_i * x_i
                    self.b += self.lr * y_i
                    updated = True

            # 全部分类正确 提前结束
            if not updated:
                break

    # 预测
    def predict(self, X):
        return np.array([self.sign(np.dot(self.w, x) + self.b) for x in X])


if __name__ == '__main__':

    # 与门(AND)数据
    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    y = np.array([-1, -1, -1, 1])

    # 创建感知机
    model = Perceptron(2, 0.1, 100)

    # 训练
    model.fit(X, y)

    # 预测
    print('权重w:',model.w)
    print('偏置b: %.1f' % model.b)
    print(f'预测结果: {model.predict(X)}')
