'''
 Python 代码演示向量数据库（Chroma），完整走一遍：文本→向量→入库→查询
'''
import chromadb
import numpy as np
# from sentence_transformers import SentenceTransformer


# ==============================
# 本地随机生成向量（不用 Hugging Face！）
# 模拟 Transformer 输出的 Embedding 向量
# ==============================
def get_embedding(text, dim = 32):
    '''
    本地生成固定随机向量，模拟词嵌入
    '''
    np.random.seed(hash(text) % 10 ** 8)
    return np.random.rand(dim).tolist()


# 1.创建本地向量库
client = chromadb.PersistentClient(path="./my_local_vector_db")
collection = client.get_or_create_collection(name="course_info")

# 2.添加文本 + 存入向量
docs = [
    "我爱水课",
    "水课很轻松，作业少",
    "我喜欢简单的课程",
    "专业课难度很大"
]

ids = [f"doc_{i}" for i in range(len(docs))]
embeddings = [get_embedding(doc) for doc in docs]

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=ids
)


# 3.语义查询（核心！）
query = "有没有轻松的课"
query_emb = get_embedding(query)

results = collection.query(
    query_embeddings=[query_emb],
    n_results=2
)


# 4.输出结果
print('查询：', query)
print('\n最相关的内容：')
for doc in results['documents'][0]:
    print(f'- {doc}')
