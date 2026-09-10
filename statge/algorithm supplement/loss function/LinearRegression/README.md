# 线性回归损失函数



# 均方误差损失函数(Mean Squared Error)L2 Loss
    Loss(y_pred, y_true) = 1/2 * (y_pred - y_true) ** 2
    其中y_pred 是模型预测值，y_true 是真实值


# 绝对值损失函数(Mean Absolute Error)L1 Loss
    Loss(y_pred, y_true) = |y_pred - y_true|
    其中y_pred 是模型预测值，y_true 是真实值


# Huber Loss
    Loss(y_pred, y_true, δ) = 1/2 * (y_pred - y_true) ** 2          if |y_pred - y_true| <= δ
    Loss(y_pred, y_true, δ) = δ * (|y_pred - y_true| - δ * 0.5)     if |y_pred - y_true| > δ
    其中δ 是一个超参数（通常取 1.0），控制"小误差"和"大误差"的分界线




# Tips
    TensorFlow/Keras 和 Scikit-learn 选择了 y_true, y_pred （真实值在前）。
    PyTorch 则选择了 y_pred, y_true （预测值在前）。
    不过，这个规律在 PyTorch 内部也有例外。例如，当你使用 torch.nn.functional.binary_cross_entropy 时，它的参数顺序是 (input, target)，也就是 (y_pred, y_true) 。但在更早的 nn.BCELoss 中，又是 (input, target)。所以，PyTorch 的底层 functional 接口倾向于 y_pred, y_true，但 nn.Module 的封装类通常用 input, target 来命名，这是一种更通用的表述。
