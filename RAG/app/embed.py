"""向量化与索引读写。

职责边界只有两件事：
1. vectorize —— 把文本块变成 TF-IDF 矩阵
2. save_index / load_index —— 索引的持久化与读取

流水线编排（parse → clean → chunk → vectorize → save）在 build_index.py，
本文件不 import parse/chunk，避免把「存储」和「数据加工」耦在一起。
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_PATH = Path(__file__).resolve().parent.parent / 'data' / 'index.pkl'


def vectorize(chunks: list[dict]) -> tuple[TfidfVectorizer, np.ndarray]:
    """把 chunks 里的 text 字段变成一个 TF-IDF 矩阵。

    返回：
        vectorizer：拟合好的 TfidfVectorizer，检索时对 query 做 transform
        matrix：    shape=(n_chunks, n_vocab) 的稠密矩阵

    中文必须用字符 n-gram：sklearn 默认的 token_pattern=r'(?u)\\b\\w\\w+\\b'
    按空白和 ASCII 词边界切词，整句中文会被当成一个 token，导致纯中文查询
    零命中（实测 recall@1 从 0.357 提升到 0.929，差距全部来自这一处）。

    已知代价，如实记录：
    - 只做字面共现，不懂同义词、不敏感于语序
    - 2-3 gram 组合数远多于词袋，维度膨胀
    - max_features=8000 是硬上限，真实 n-gram 数超过 8000 时会**静默截断**
      低频项。构建时打印实际词表大小，用来判断是否被截断

    为什么返回稠密矩阵：W07 阶段要在 index.py 里逐行手写余弦、把计算过程
    摊开演示；代价是内存（101 x 8000 x 8B ≈ 6.6MB）和 101 次 Python 循环。
    后续应改为稀疏矩阵 + sklearn.metrics.pairwise.cosine_similarity。
    """
    texts = [c['text'] for c in chunks]
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), max_features=8000)
    matrix = vectorizer.fit_transform(texts).toarray()
    return vectorizer, matrix


def save_index(doc_id: str, vectorizer: TfidfVectorizer, matrix: np.ndarray,
               chunks: list[dict]) -> Path:
    """把索引写到 INDEX_PATH，返回文件路径。

    重复构建的行为：同一文档、同一份代码，块集与向量矩阵都是确定的，
    因此重复运行不会产生重复块，也不会让索引漂移。
    但 `created` 是构建时间戳，每次都会变，文件字节并不相同——
    不要用文件哈希判断「索引是否需要重建」，要看构建参数和源文档是否变化。
    """
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'doc_id': doc_id,
        'vectorizer': vectorizer,
        'matrix': matrix,
        'chunks': chunks,
        'created': int(time.time()),
    }
    INDEX_PATH.write_bytes(pickle.dumps(data))
    return INDEX_PATH


def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f'索引文件不存在：{INDEX_PATH}\n'
            f'请先运行：python RAG/app/build_index.py'
        )
    return pickle.loads(INDEX_PATH.read_bytes())
