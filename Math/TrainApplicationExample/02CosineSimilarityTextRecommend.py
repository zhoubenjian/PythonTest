'''
余弦相似度文本推荐
'''
import numpy as np


# 1.准备语料——5 句中文，包含 AI 学习与日常生活两类话题
sentences = [
    "我喜欢用python学习机器学习",
    "python是学习人工智能的好工具",
    "今天天气很好适合出去散步",
    "深度学习需要大量的数据和算力",
    "散步是一种很好的放松方式",
]


# 2.构建词表（教学简化版，手动定义可能出现的词语）
vocab_words = [
    "我", "喜欢", "用", "python", "学习", "机器学习", "是",
    "人工智能", "好", "工具", "今天", "天气", "很", "适合",
    "出去", "散步", "深度学习", "需要", "大量", "的", "数据",
    "算力", "一种", "放松", "方式"
]


def to_vector(sentence, vocab):
    """
    将文本转换为向量表示 出现过的词标记为1
    :param sentence: 输入文本
    :param vocab: 词表
    :return: 向量表示
    """
    vec = np.zeros(len(vocab))
    for i, w in enumerate(vocab):
        if w in sentence:
            # 词出现了就记1
            vec[i] = 1
    return vec

# 5个句子 => 5个向量化
vectors = np.array([to_vector(s, vocab_words) for s in sentences])
print("每句话的向量维度:", vectors.shape[1], "(词表大小)", sep='')


# 3.手写余弦相似度函数（不调 sklearn）
def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度
    :param a: 向量1
    :param b: 向量2
    :return: 余弦相似度
    """
    dot = np.dot(a, b)
    # a模长
    norm_a = np.linalg.norm(a)
    # b模长
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# 4.给定查询句子，找出最相似的句子
query = "我在学习python和人工智能"
query_vec = to_vector(query, vocab_words)

sims = [cosine_similarity(query_vec, v) for v in vectors]

print(f"\n查询句子: {query!r}\n")
for s, sim in sorted(zip(sentences, sims), key=lambda x: -x[1]):
    print(f"相似度 {sim:.3f}  <=  {s}")


# 5.验证：不相关的句子相似度为 0
print("\n验证：两个完全不相关的句子")
v1 = to_vector("python机器学习", vocab_words)
v2 = to_vector("散步放松", vocab_words)
print(f"'python机器学习' vs '散步放松': "
      f"cos = {cosine_similarity(v1, v2):.3f}")
