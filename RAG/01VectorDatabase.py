'''
实现本地向量库
'''
import os
import chromadb
from sentence_transformers import SentenceTransformer


# 1.准备你的本地知识库
knowledge = [
    "什么是AI：人工智能（AI）是一门使机器模拟人类智能的技术。",
    "什么是RAG：RAG代表检索增强生成，通过检索外部知识提升大模型回答准确性。",
    "本地RAG消耗Token吗：本地RAG不调用云端API，不消耗Token，完全免费。",
    "FAISS是什么：FAISS是Facebook开源的向量检索库，用于本地高效检索。",
    "RAG的作用：RAG让大模型能引用外部知识，避免胡说八道。"
]


# 2.加载本地路径的Embedding模型
local_model_path = "../local_models/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(local_model_path, device="cpu")

# 将知识库文本转换成向量
vectors = embed_model.encode(knowledge)


# 3.初始化一个持久化的Chroma客户端
client = chromadb.PersistentClient(path="./local_chroma_db")


# 4.获取或创建一个”knowledge_base“的集合
collection = client.get_or_create_collection(name="knowledge_base")


# 5.为每一条数据生成唯一ID，并将文本，向量添加到collection中
ids = [f'doc_{i}' for i in range(len(knowledge))]
collection.add(
    documents=knowledge,
    embeddings=vectors.tolist(),
    ids=ids
)


# 6.定义查询函数
def search(query_key):
    # 问题向量化
    query_vector = embed_model.encode(query_key)
    # 在chroma中查询最相似的1条记录
    results = collection.query(
        n_results=1,
        query_embeddings=query_vector.tolist()
    )
    # 打印检索到的答案
    best_answer = results['documents'][0][0]
    print(f'问题：{query_key}')
    print(f'答案：{best_answer}')


# 7.执行一次查询
if __name__ == '__main__':
    search("RAG的作用？")




