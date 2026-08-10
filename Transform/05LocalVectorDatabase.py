import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer


# 本地向量化模型
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")    # 支持中文

# 磁盘持久化向量库
client = chromadb.PersistentClient(path="./chroma_db")

# 获取或创建集合
try:
    coll = client.create_collection("local_demo")
except:
    coll = client.get_collection("local_demo")

# 测试文本
docs = [
    "今天天气很好",
    "阳光明媚，适合出门",
    "股市大涨了",
    "明天可能会下雨"
]
ids = [f"doc_{i}" for i in range(len(docs))]
embeds = model.encode(docs).tolist()

# 写入向量库
coll.add(documents=docs, embeddings=embeds, ids=ids)

# 语义检索
query = "今天天气怎么样？"
q_embed = model.encode([query]).tolist()
res = coll.query(query_embeddings=q_embed, n_results=2)

print("查询问题：", query)
print("匹配结果：")
for idx, doc in enumerate(res["documents"][0]):
    print(f"{idx+1}. {doc}，相似度距离：{res['distances'][0][idx]}")