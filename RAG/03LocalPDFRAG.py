'''
原生 Chroma API，不依赖 LangChain 高层封装
'''
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# 配置
PERSIST_PATH = "./local_pdf_chroma_db_02"   # 本地向量库


'''
1.本地embedding
'''
embedding = SentenceTransformerEmbeddingFunction(
    model_name = "../local_models/bge-small-zh-v1.5",
    device = "cpu"
)

client = chromadb.PersistentClient(path = PERSIST_PATH)
collections = client.get_or_create_collection(
    name = "pdf_local_rag",
    embedding_function= embedding
)


'''
2.读取pdf
'''
reader = PdfReader("./pdf/AlibabaJavaDevelopmentManual.pdf")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 80
)

all_chunks = []
metas = []
ids = []

for page_idx, page in enumerate(reader.pages):
    page_text = page.extract_text()
    if not page_text:
        continue
    chunks = text_splitter.split_text(page_text)
    for c_idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metas.append({"page": page_idx})
        ids.append(f"page_{page_idx}_chunk_{c_idx}")

collections.add(
    documents = all_chunks,
    metadatas = metas,
    ids = ids
)


'''
查询
'''
query_result = collections.query(
    query_texts = ["变量命名规范"],
    n_results = 3
)

print("召回文本：")
for doc, meta, dist in zip(
    query_result["documents"][0],
    query_result["metadatas"][0],
    query_result["distances"][0]
):
    print(f"page:{meta['page']}, distance:{dist:.4f}")
    print(doc)


