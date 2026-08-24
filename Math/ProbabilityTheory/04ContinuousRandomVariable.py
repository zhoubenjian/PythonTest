'''
连续随机变量（Continuous Random Variable）
'''
import numpy as np


# 正态分布 N(0,1)
samples = np.random.normal(0, 1, size=10000)
print(f"正态分布N(0,1) 采样: 均值={samples.mean():.3f}, 标准差={samples.std():.3f}")

print()

# 68-95-99.7 法则
for k in [1, 2, 3]:
    within = np.mean(np.abs(samples) < k)
    theory = {1:0.6827, 2:0.9545, 3:0.9973}[k]
    print(f"μ±{k}σ: {within:.4f} (理论 {theory:.4f})")

# 中心极限定理：12个骰子和 ≈ 正态
dice = np.random.randint(1,7,(100000,12)).sum(axis=1)
print(f"\n12个骰子和: 均值={dice.mean():.1f}(理论42), 接近正态")