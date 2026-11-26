"""向量化与索引持久化。

W07 任务：用 TF-IDF（sklearn）把文本块变成向量，
并实现「重复导入幂等」——同一批块多次运行只生成一份索引。
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from parse import PDF_PATH
from chunk import DOC_ID, build_chunks, clean_pages, find_repeated_lines, load_pages

INDEX_PATH = Path(__file__).resolve().parent.parent / 'data' / 'index.pkl'


def vectorize(chunks: list[dict]) -> tuple[TfidfVectorizer, np.ndarray]:
    """把 chunks 里的 text 字段变成 TF-IDF 矩阵。

    返回：
        vectorizer：拟合好的 TfidfVectorizer（检索时对查询词 transform 用）
        matrix：    稠密化后的矩阵，shape=(n_chunks, n_vocab)

    为什么稠密：101 块、词汇量几千，稀疏矩阵存 pickle 也方便，
    但稠密做余弦相似度更直观，W07 验收里我们要手动算 cosine。
    """
    texts = [c['text'] for c in chunks]
    # 中文没有空格分词，sklearn 默认 token_pattern[r'(?u)\b\w\w+\b'] 会把
    # 整句中文当成一个 token，导致纯中文查询零命中。改用字符 n-gram：
    # char_wb 按字切，2-3 gram 能覆盖「流程控制」「条件控制」这类词。
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), max_features=8000)
    matrix = vectorizer.fit_transform(texts).toarray()
    return vectorizer, matrix


def build_and_save() -> Path:
    """完整流程：load → parse → chunk → vectorize → persist。

    幂等性：chunk 的 id 用 doc_id+page+order 哈希得出，相同块集两次
    运行得到完全相同的向量矩阵（TF-IDF 词表顺序确定），因此不需要
    额外检查——覆盖写就等于"没变化"。
    """
    raw = load_pages(PDF_PATH)
    noise = find_repeated_lines(raw)
    pages = clean_pages(raw, noise)
    chunks = build_chunks(pages)
    vectorizer, matrix = vectorize(chunks)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'doc_id': DOC_ID,
        'vectorizer': vectorizer,
        'matrix': matrix,
        'chunks': chunks,
        'created': int(time.time()),
    }
    INDEX_PATH.write_bytes(pickle.dumps(data))
    print(f'索引已保存：{INDEX_PATH}')
    print(f'块数={len(chunks)}，维度={matrix.shape[1]}，大小={INDEX_PATH.stat().st_size // 1024}KB')
    return INDEX_PATH


def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f'索引文件不存在：{INDEX_PATH}，请先运行 index.py')
    return pickle.loads(INDEX_PATH.read_bytes())


if __name__ == '__main__':
    build_and_save()
