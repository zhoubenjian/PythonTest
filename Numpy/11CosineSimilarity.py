import numpy as np


# 手动模拟32维向量
def get_vec(text):
    np.random.seed(hash(text) % 10000)
    return np.random.randn(32)

# 文本
query = get_vec("有没有轻松的课？")
a = get_vec("水课很轻松，作业少")
b = get_vec("我喜欢简单的课程")
c = get_vec("专业课难度很大")

# 余弦相似度
def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print("query vs 水课:", cos_sim(query, a))
print("query vs 简单课程:", cos_sim(query, b))
print("query vs 专业课:", cos_sim(query, c))


print('\n' * 5)


# 求夹角
a = np.array([1, 0])
b = np.array([2, 1])
print(np.linalg.norm(a))
print(np.linalg.norm(b))
print(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
