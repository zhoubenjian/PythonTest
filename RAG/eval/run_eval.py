"""检索基线评测：用 questions.json 跑 TF-IDF 检索，记录 Recall@k 与 MRR。

用法（在 RAG/eval 目录下，或直接在仓库根目录跑）：
    python RAG/eval/run_eval.py

产物：
    baseline_tfidf.json —— 可作为 W10 换检索方案（dense embedding / BM25 / 重排）
    后的对照基线，所有数字必须来自同一条命令、同一份索引。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'app'))

from index import search  # noqa: E402

QUESTIONS = BASE / 'questions.json'
REPORT = BASE / 'baseline_tfidf.json'


def main(top_k: int = 5) -> None:
    data = json.loads(QUESTIONS.read_text(encoding='utf-8'))
    answerable = [q for q in data['questions'] if q['type'] != 'unanswerable']
    unanswerable = [q for q in data['questions'] if q['type'] == 'unanswerable']

    details = []
    reciprocal_ranks = []

    for q in answerable:
        t0 = time.perf_counter()
        hits = search(q['question'], top_k=top_k)
        cost_ms = round((time.perf_counter() - t0) * 1000, 1)
        pages = [h.page for h in hits]
        want = set(q['source_pages'])
        rank = next((i + 1 for i, p in enumerate(pages) if p in want), None)
        if rank:
            reciprocal_ranks.append(1.0 / rank)
        details.append({
            'id': q['id'],
            'type': q['type'],
            'question': q['question'],
            'expected_pages': q['source_pages'],
            'hit_pages': pages,
            'hit_scores': [round(h.score, 4) for h in hits],
            'first_hit_rank': rank,
            'cost_ms': cost_ms,
        })

    for q in unanswerable:
        hits = search(q['question'], top_k=top_k)
        details.append({
            'id': q['id'],
            'type': q['type'],
            'question': q['question'],
            'expected_pages': [],
            'hit_pages': [h.page for h in hits],
            'hit_scores': [round(h.score, 4) for h in hits],
            'first_hit_rank': None,
            'top1_score': round(hits[0].score, 4) if hits else 0.0,
        })

    n = len(answerable)
    answerable_details = [d for d in details if d['type'] != 'unanswerable']
    unanswerable_details = [d for d in details if d['type'] == 'unanswerable']

    summary = {
        'answerable_count': n,
        'unanswerable_count': len(unanswerable),
        'top_k': top_k,
        'recall@1': round(sum(1 for d in answerable_details if d['first_hit_rank'] == 1) / n, 4),
        f'recall@{top_k}': round(sum(1 for d in answerable_details if d['first_hit_rank']) / n, 4),
        'mrr': round(sum(reciprocal_ranks) / n, 4),
        'avg_top1_score_answerable': round(
            sum(d['hit_scores'][0] if d['hit_scores'] else 0.0 for d in answerable_details) / n, 4
        ),
        'max_top1_score_unanswerable': round(
            max((d['top1_score'] for d in unanswerable_details), default=0.0), 4
        ),
    }

    report = {
        'meta': {
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'retriever': 'tfidf(max_features=5000, ngram=(1,2)) + cosine',
            'doc_id': data['meta']['doc_id'],
            'note': '命中判定为 page 级：Top-k 中任一块的 page 属于 source_pages 即算命中',
        },
        'summary': summary,
        'details': details,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"{'ID':<5}{'类型':<14}{'期望页':<10}{'命中页':<16}{'名次':<6}问题")
    print('-' * 100)
    for d in details:
        rank = d['first_hit_rank'] if d['first_hit_rank'] else '-'
        print(f"{d['id']:<5}{d['type']:<14}{str(d['expected_pages']):<10}"
              f"{str(d['hit_pages']):<16}{str(rank):<6}{d['question'][:36]}")

    print('-' * 100)
    for k, v in summary.items():
        print(f'{k}: {v}')
    print(f'\n报告已保存：{REPORT}')


if __name__ == '__main__':
    main()
