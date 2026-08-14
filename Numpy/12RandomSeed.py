import numpy as np


# 未指定seed 每次生成随机数不一致
for i in range(5):
    print(np.random.rand(20))
    print('-' * 50)


print('\n' + '=' * 77 + '\n')


for i in range(5):
    np.random.seed(42)          # 固定输出
    print(np.random.rand(20))   # 生成结果一样
    print('-' * 50)