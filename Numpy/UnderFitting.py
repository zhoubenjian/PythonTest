'''
欠拟合：
    模型过于简单，无法捕捉数据中的基本规律或模式。就像一个学生只学了加法，却要去解微积分题目。

    表现：模型在训练数据上表现就很差（例如，准确率低，误差大）。

    原因：模型复杂度太低，特征不足，或训练不充分。

    类比：用一条直线（一次多项式）去拟合有明显弯曲趋势的数据。
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


np.random.seed(42)
X = np.linspace(0, 10, 20)
y_true = np.sin(X)                      # 真实的潜在规律（我们不知道）
y_noise = np.random.randn(20) * 0.3     # 随机噪声
y = y_true + y_noise                    # 我们实际观测到的数据

# 尝试用1阶多项式（直线）拟合
poly = PolynomialFeatures(degree = 1)
X_poly1 = poly.fit_transform(X.reshape(-1, 1))
model_under = LinearRegression()
model_under.fit(X_poly1, y)
y_pred_under = model_under.predict(X_poly1)

mse_train_under = mean_squared_error(y, y_pred_under)
print(f'欠拟合模型在训练集上的均方误差 (MSE): {mse_train_under:.4f}')


print('=' * 66)


# 尝试用3阶多项式拟合
poly = PolynomialFeatures(degree = 3)
X_poly3 = poly.fit_transform(X.reshape(-1, 1))
model_good = LinearRegression()
model_good.fit(X_poly3, y)
y_pred_good = model_good.predict(X_poly3)

mse_train_good = mean_squared_error(y, y_pred_good)
print(f'良好拟合模型在训练集上的均方误差 (MSE): {mse_train_good:.4f}')