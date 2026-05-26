import numpy as np


# 余弦相似度：衡量方向相似性
def cosine_similar(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 欧氏距离：衡量绝对位置差异
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# 点积
def dot_product(a, b):
    return np.dot(a, b)


# 示例向量
v1 = np.array([0.12, -0.54, 0.87, 0.03])
v2 = np.array([0.10, -0.50, 0.90, 0.05])
v3 = np.array([-0.80, 0.20, -0.30, 0.70])

print(f'v1与v2的余弦相似度：{cosine_similar(v1, v2):.4f}')
print('v1与v3的余弦相似度：%.4f' % (cosine_similar(v1, v3)))
