'''
向量余弦相似度 <=> 向量夹角的余弦值
    单位向量：只留方向，L2模长为1
        任何非零向量除以自己的模长，得到长度为 1 的 单位向量；
        向量归一化：把一个向量“缩放”成单位长度（即模长为 1），同时保持它原本的方向不变。

    余弦相似度：只看方向，不看大小
        1 表示完全相似，-1 表示完全相反，0 表示不相关；

    a⋅b=|a|*|b|*cosθ <=> cosθ=a⋅b/|a|*|b|
'''
import numpy as np


a = np.array([3, 4, 0])
b = np.array([6, 8, 0])

# 模长
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print(f'|a| = {norm_a:.1f}, |b| = {norm_b:.1f}')    # |a| = 5.0, |b| = 10.0

# 单位向量（归一化）
unit_a = a / norm_a
print('a 归一化：', unit_a)    # a 归一化：[0.6, 0.8, 0.0]
print(np.linalg.norm(unit_a))    # |unit_a| = 1.0
unit_b = b / norm_b
print('b 归一化：', unit_b)    # b 归一化：[0.6, 0.8, 0.0]
print(np.linalg.norm(unit_b))    # |unit_b| = 1.0


print('\n' + '*' * 30 + '\n')


cos_sim = a @ b / (norm_a * norm_b)
print('余弦相似度：{:.4f}'.format(cos_sim))       # 余弦相似度：1.0000


print('\n' + '*' * 30 + '\n')


# 批量计算余弦相似度矩阵
embeddings = np.array([
    [0.2, 0.5, 0.1, 0.8, 0.3],
    [0.3, 0.6, 0.2, 0.7, 0.4],
    [0.9, 0.1, 0.8, 0.1, 0.9],
])
# 计算每个向量的 L2 范数（模长）
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

# 归一化
normalized = embeddings / norms
print('归一化后的向量：\n', normalized)

# 计算余弦相似度矩阵
sim_matrix = normalized @ normalized.T
print('\n余弦相似度矩阵：')
'''
[[1.    0.978 0.431]
 [0.978 1.    0.571]
 [0.431 0.571 1.   ]]
'''
print(np.round(sim_matrix, 3))







