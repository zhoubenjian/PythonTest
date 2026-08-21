import numpy as np


# 固定种子
rng_fixed = np.random.RandomState(42)
for i in range(10):
    print(rng_fixed.randint(0, 10), end=', ')


print('\n' * 2 + '=' * 44 + '\n')


# 真正随机（默认使用系统时间）
rng_random = np.random.RandomState()
for i in range(10):
    print(rng_random.choice(['红 (Red)', '橙 (Orange)', '黄 (Yellow)', '绿 (Green)', '蓝 (Blue)', '靛 (Indigo)', '紫 (Violet)']), end=', ')
