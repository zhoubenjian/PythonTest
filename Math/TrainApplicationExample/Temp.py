'''
模拟基于余弦相似度的文本推荐
'''
import numpy as np
import jieba
import re


# 1.初始化文本
sentences = [
    '实现中华民族伟大复兴',
    '依法治国，有法可依，有法必依，执法必严',
    '公平、公正、公开',
    '文明其精神，野蛮其体魄',
    '德智体美劳全面发展'
]


# 2.切分中文分词
def tokenize(text):
    """
    中文分词（正则匹配去除非中文字符，再切分分词）
    :param text: 输入文本
    :return: 分词后的列表(返回值是list)
    """
    return list(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', text)))

token_list = [tokenize(s) for s in sentences]
print(token_list)


# 3.遍历嵌套列表，提取所有词元，然后去重
flat_tokens = [word for sentence in token_list for word in sentence]
unique_tokens = np.unique(flat_tokens)
print(f'\n去重后的词元：\n{unique_tokens}')


# 4.将每个句子转化为词频向量（Bag of Words）
# 将 unique_tokens 转为普通列表
token_list_unique = unique_tokens.tolist()
vectors = []
for tokens in token_list:
    vec = np.zeros(len(unique_tokens))
    for word in tokens:
        if word in unique_tokens:
            # 在计算余弦相似度时，强烈建议使用 += 1（累加模式）
            vec[token_list_unique.index(word)] += 1
    vectors.append(vec)

# 将收集好的向量列表转换为 NumPy 矩阵
vectors = np.array(vectors)


# 5.手动计算计算余弦相似度
def cosine_similarity(v1, v2):
    """
    余弦相似度
    :param v1: 向量1
    :param v2: 向量2
    :return: 余弦相似度
    """
    return np.round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 4)

input_text = '伟大复兴'
input_tokens = tokenize(input_text)
# 将输入文本转化为与语料库同维度的向量
input_vec = np.zeros(len(unique_tokens))
for word in input_tokens:
    if word in unique_tokens:
        input_vec[token_list_unique.index(word)] += 1

# 计算输入文本与所有句子的相似度
print("-" * 66)
for i, vec in enumerate(vectors):
    sim = cosine_similarity(input_vec, vec)
    print(f"与句子 '{sentences[i]}' 的相似度: {sim}")
