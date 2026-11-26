"""向量检索：输入查询词，返回 Top-k 来源块。

W07 验收要求：返回文本 + 来源，重复导入无重复。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embed import load_index


@dataclass
class Hit:
    id: str
    score: float
    text: str
    page: int
    order: int


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a,b)，手动写一遍而不是用 sklearn.metrics.pairwise.cosine_similarity，
    因为 W07 验收要求能讲清余弦相似度是什么。"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def search(query: str, top_k: int = 3) -> list[Hit]:
    data = load_index()
    vectorizer: object = data['vectorizer']
    matrix: np.ndarray = data['matrix']
    chunks: list[dict] = data['chunks']

    q = vectorizer.transform([query]).toarray()[0]
    scores = np.array([_cosine(q, row) for row in matrix])
    top_idx = np.argsort(-scores)[:top_k]
    return [
        Hit(
            id=chunks[i]['id'],
            score=float(scores[i]),
            text=chunks[i]['text'],
            page=chunks[i]['page'],
            order=chunks[i]['order'],
        )
        for i in top_idx
        if scores[i] > 0
    ]


if __name__ == '__main__':
    # 验收块：用两道评测题验证检索能命中对应页
    cases = [
        '重写 equals 必须重写什么方法',     # 预期 page 12
        'finally 块里能写 return 吗',       # 预期 page 21
        '表名不能用什么形式的名词',          # 预期 page 27
    ]
    for q in cases:
        hits = search(q, top_k=2)
        print(f'\nQ: {q}')
        for h in hits:
            print(f'  [{h.score:.3f}] page={h.page} id={h.id[:10]}... {h.text[:50]}...')
        if not hits:
            print('  (无命中)')
