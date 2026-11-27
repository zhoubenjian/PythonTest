"""索引构建流水线：parse → clean → chunk → vectorize → 持久化。

单独成文件的原因：embed.py 只负责「向量化 + 存取」，
对 parse/chunk 的依赖只出现在这里，职责边界清楚。
"""
from __future__ import annotations

from chunk import DOC_ID, build_chunks, clean_pages, find_repeated_lines, load_pages
from embed import save_index, vectorize
from parse import PDF_PATH

VOCAB_CAP = 8000


def build() -> None:
    raw = load_pages(PDF_PATH)
    noise = find_repeated_lines(raw)
    pages = clean_pages(raw, noise)
    chunks = build_chunks(pages)
    vectorizer, matrix = vectorize(chunks)
    path = save_index(DOC_ID, vectorizer, matrix, chunks)

    vocab_size = len(vectorizer.vocabulary_)
    truncated = '已触顶，存在低频 n-gram 被截断' if vocab_size >= VOCAB_CAP else '未截断'
    print(f'索引已保存：{path}')
    print(f'块数={len(chunks)}，词表={vocab_size}（上限 {VOCAB_CAP}，{truncated}），'
          f'维度={matrix.shape[1]}，大小={path.stat().st_size // 1024}KB')


if __name__ == '__main__':
    build()
