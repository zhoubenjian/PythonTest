'''
可视化对比不同损失函数的性能
'''
import numpy as np
import matplotlib.pyplot as plt

import common.mpl_config


# 模拟误差范围[-5, 5]
errors = np.linspace(-5, 5, 500)

# 均绝对误差（L1 Loss）
mae_loss = np.abs(errors)

# 均平方误差（L2 Loss）
mse_loss = 1/2 * errors ** 2

# Huber损失函数
δ = 1.0
huber_loss = np.where(np.abs(errors) <= δ, 1/2 * errors ** 2, δ * (np.abs(errors) - 0.5 * δ))


plt.figure(figsize=(10, 6))
plt.plot(errors, mae_loss, label='MAE(L1)', linewidth=2)
plt.plot(errors, mse_loss, label='MSE(L2)', linewidth=2)
plt.plot(errors, huber_loss, label='Huber(δ=1)', linewidth=2, linestyle='--')
plt.xlabel('误差 (y - ŷ)')
plt.ylabel('损失值')
plt.title('回归损失函数对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 10)
plt.show()



