"""W06-1：PDF 解析

目标：把 PDF 变成「带页码的文本页」，并识别两类常见脏数据：
  1. 页眉页脚——在几乎每一页重复出现的短行，留着会污染检索
  2. 无文字层——扫描件，提取出来是空字符串，必须显式报错

产出结构：每页一个 dict
    {'page': 1, 'text': '本页正文（已剔除页眉页脚）'}

验收（__main__ 里已写好，别改）：
  - 页数与 PDF 实际页数一致
  - 有文字的页不少于 80%（第 1 页为封面，天然为空）
  - 扫描件能被 assert_text_layer 拦下
"""
from collections import Counter
from pathlib import Path

import pymupdf

PDF_PATH = Path(__file__).resolve().parent.parent / 'pdf' / 'AlibabaJavaDevelopmentManual.pdf'


def load_pages(pdf_path):
    """解析 PDF，返回 [{'page': int, 'text': str}]，page 从 1 开始。"""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text('text')
        pages.append({'page': i + 1, 'text': text})
    doc.close()
    return pages


def find_repeated_lines(pages, min_ratio=0.6):
    """找出在超过 min_ratio 比例的页面里都出现的短行，视为页眉页脚。"""
    counter = Counter()
    for p in pages:
        lines = set(line.strip() for line in p['text'].split('\n') if line.strip())
        for line in lines:
            counter[line] += 1
    threshold = len(pages) * min_ratio
    return {line for line, cnt in counter.items() if cnt >= threshold and len(line) <= 80}


def clean_pages(pages, noise_lines):
    """把页眉页脚从每页文本里剔除，返回新的 pages。"""
    result = []
    for p in pages:
        lines = p['text'].split('\n')
        cleaned = [line for line in lines if line.strip() not in noise_lines]
        result.append({'page': p['page'], 'text': '\n'.join(cleaned)})
    return result


def assert_text_layer(pages, min_avg_chars=20):
    """确认 PDF 有文字层，否则抛 ValueError（用于识别扫描件）。"""
    total = sum(len(p['text']) for p in pages)
    avg = total / len(pages) if pages else 0
    if avg < min_avg_chars:
        raise ValueError(f'文字层平均每页 {avg:.1f} 字符，低于阈值 {min_avg_chars}，可能是扫描件')


if __name__ == '__main__':
    raw = load_pages(PDF_PATH)
    print(f'总页数: {len(raw)}')

    noise = find_repeated_lines(raw)
    print(f'疑似页眉页脚 {len(noise)} 行，例如: {sorted(noise)[:5]}')

    pages = clean_pages(raw, noise)
    assert_text_layer(pages)

    # ===== 验收，不要改 =====
    doc = pymupdf.open(PDF_PATH)
    assert len(pages) == doc.page_count, f'页数不一致: {len(pages)} vs {doc.page_count}'
    non_empty = sum(1 for p in pages if p['text'].strip())
    assert non_empty >= len(pages) * 0.8, f'有文字的页仅 {non_empty}/{len(pages)}，可能是扫描件'
    assert all(p['page'] == i + 1 for i, p in enumerate(pages)), '页码未从 1 连续编号'

    # 扫描件必须被拦下
    try:
        assert_text_layer([{'page': 1, 'text': ''}])
    except ValueError:
        pass
    else:
        raise AssertionError('空文字层应当抛 ValueError')

    print(f'--- 清洗后第 2 页前 300 字（第 1 页是封面，天然为空）---')
    print(pages[1]['text'][:300])
    print('验收 PASS')