"""W06-1：PDF 解析

目标：把 PDF 变成「带页码的文本页」，并识别两类常见脏数据：
  1. 页眉页脚——在几乎每一页重复出现的短行，留着会污染检索
  2. 无文字层——扫描件，提取出来是空字符串，必须显式报错

产出结构：每页一个 dict
    {'page': 1, 'text': '本页正文（已剔除页眉页脚）'}

验收（__main__ 里已写好，别改）：
  - 页数与 PDF 实际页数一致
  - 首页 text 非空
  - 扫描件能被 assert_text_layer 拦下
"""
from collections import Counter
from pathlib import Path

import pymupdf

PDF_PATH = Path(__file__).resolve().parent.parent / 'pdf' / 'AlibabaJavaDevelopmentManual.pdf'


def load_pages(pdf_path):
    """解析 PDF，返回 [{'page': int, 'text': str}]，page 从 1 开始。

    注意：page 用 1 起算，因为人读页码和引用都从 1 开始，
    避免后面每次展示来源都 +1 转换（少一次转换就少一次错）。
    """
    # TODO 1: 用 pymupdf.open 打开，遍历每一页取纯文本
    #   提示：page.get_text('text') 是纯文本模式
    #   注意：本 PDF 第 1 页是封面，提出来是空的，这是正常情况，不要当异常处理
    # TODO 2: 拼成 [{'page': i + 1, 'text': ...}]
    raise NotImplementedError('练习：实现 load_pages')


def find_repeated_lines(pages, min_ratio=0.6):
    """找出在超过 min_ratio 比例的页面里都出现的短行，视为页眉页脚。

    做法要点：
      - 逐页按行切分，去掉空白行
      - 统计整行文本的出现页数（注意：同一页里重复出现只算一次）
      - 只保留「长度较短」的行，避免把正文长句误判
    返回：{行文本} 的集合
    """
    # TODO 3: 实现频次统计
    #   提示：Counter 统计「行 -> 出现页数」时，要先用 set 把每页的行去重
    raise NotImplementedError('练习：实现 find_repeated_lines')


def clean_pages(pages, noise_lines):
    """把页眉页脚从每页文本里剔除，返回新的 pages。"""
    # TODO 4: 按行过滤，保留非噪声行；保持返回结构与输入一致
    raise NotImplementedError('练习：实现 clean_pages')


def assert_text_layer(pages, min_avg_chars=20):
    """确认 PDF 有文字层，否则抛 ValueError（用于识别扫描件）。"""
    # TODO 5: 算平均每页字符数，低于阈值就抛，错误信息里带上实际数字
    raise NotImplementedError('练习：实现 assert_text_layer')


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
