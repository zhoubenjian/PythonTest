import numpy as np

w = np.array([3, -4])

# ------ L1范数 ------
l1_norm = np.linalg.norm(w, ord=1)
# 等价手工计算：|3| + |-4|
l1_manual = abs(w[0]) + abs(w[1])
print(f"L1范数(曼哈顿): {l1_norm}, 手工={l1_manual}")

# ------ L2范数 ------
l2_norm = np.linalg.norm(w, ord=2)
# # 等价手工：sqrt(3² + (-4)²)
l2_manual = np.sqrt(w[0]**2 + w[1]**2)
print(f"L2范数(欧氏距离，带根号): {l2_norm}, 手工={l2_manual}")

# ------ L2范数的平方（Ridge正则用的，不开根号！） ------
l2_square = np.sum(w ** 2)
l2_square_manual = w[0]**2 + w[1]**2
print(f"L2平方（Ridge惩罚项）: {l2_square}, 手工={l2_square_manual}")
