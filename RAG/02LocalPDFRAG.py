'''
chroma + bge‑small‑zh‑v1.5（对中文友好）实现本地RAG:
    LangChain 封装版（PDF 加载 => 切分 => Chroma 持久化 => RAG 问答）
'''
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceEmbeddings    # 重点：你的版本没有langchain‑huggingface，从community导入
# 或直接从langchain_huggingface中导入HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
# ⚠️ RetrievalQA 迁移到 langchain_classic
from langchain_classic.chains import RetrievalQA


# ----------------配置----------------
EMBEDDING_MODEL_PATH = "../local_models/bge-small-zh-v1.5"  # 本地向量模型
PERSIST_DIR = "./local_pdf_chroma_db_01"
COLLECTION_NAME = "pdf_rag_collection"                      # 集合名称，必须固定
PDF_PATH = "./pdf/AlibabaJavaDevelopmentManual.pdf"


'''
1.本地Embedding bge-small-zh-v1.5
'''
embedding = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL_PATH,  # 指定向量模型
    model_kwargs = {"device": "cpu"},   # GPU改为 "cuda"
    encode_kwargs = {"normalize_embeddings": True}
)


'''
2.Chroma持久化向量库 
'''
local_pdf_embedding_db = None
# 向量库存在，且目录不为空
db_exist = os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0

if db_exist:

    print(f"✅检测到已存在向量库，直接加载：{PERSIST_DIR}")
    local_pdf_embedding_db = Chroma(
        persist_directory = PERSIST_DIR,
        embedding_function = embedding,
        collection_name = COLLECTION_NAME
    )

else:

    print(f"⚠️向量库不存在，开始新建，加载PDF：{PDF_PATH}")
    # 加载pdf
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    '''
    chunk_size：每个文本块的最大字符数（不是 token 数）
    chunk_overlap：保持上下文的连贯性，避免重要信息在切片边界处被截断丢失
    separators：优先级顺序的分隔符列表（注意：按顺序优先级递减）

    实际操作流程：
    先尝试用 \n\n 分割，看块大小是否 ≤ chunk_size
    如果不行，尝试用 \n
    依次降级，直到找到合适的分隔符
    如果所有分隔符都不满足，则强制按字符数截断
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,       # 每块最大字符数500
        chunk_overlap = 80,     # 相邻文本之间重叠字符数80
        separators = ["\n\n", "\n", ".", " ", ", "]     # 分隔符及优先级
    )
    split_docs = text_splitter.split_documents(documents)
    print(f'原始页数：{len(documents)}，拆分chunk数量：{len(split_docs)}')
    print(f"文档拆分chunk数量：{len(split_docs)}")

    local_pdf_embedding_db = Chroma.from_documents(
        documents = split_docs,
        embedding = embedding,
        persist_directory = PERSIST_DIR,
        collection_name = COLLECTION_NAME
    )
    print("✅向量库新建完成")


'''
3.构建检索器
'''
retriever = local_pdf_embedding_db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.4}
)


# ----------------------5.本地大模型（演示，可替换Ollama / vLLM）----------------------
# 注意：all‑MiniLM只是embedding，不是大模型，只做向量化；生成需要另外LLM
# 这里只是示例，实际项目建议用Ollama或者vLLM
# llm = HuggingFacePipeline(pipeline=...)

# 如果不想跑本地大模型，可以只做【检索测试】，看召回的文本是否正确
def only_retrieve(question):
    docs = retriever.invoke(question)
    print(f"\n===查询问题：{question}===")
    for idx, d in enumerate(docs):
        print(f"\n---chunk{idx+1} page:{d.metadata['page']}---")
        print(d.page_content)
    return docs


'''
4.测试检索
'''
if __name__ == "__main__":
    # 只做向量检索测试，验证all‑MiniLM召回效果
    only_retrieve("变量命名规范?")

