"""W06-2：切块、元数据与稳定 ID

三个概念先说清楚，写代码时就知道每一步为什么存在：

  切块   ：一页 1000～1500 字，直接整页塞进模型既超上下文又稀释相似度，
           所以要切成小段。切点必须落在语义边界（段落 > 句子），
           切在句子中间是最典型的错误。
  元数据 ：块必须知道自己从哪来（doc_id + page），否则回答无法给出处。
  稳定 ID：同一份文档重新导入，块 ID 必须不变。否则重复导入就变成重复索引。

产出结构：
    {'id': 'a1b2c3d4e5f6', 'doc_id': 'ali_java_manual',
     'page': 3, 'order': 2, 'text': '...'}

验收（__main__ 里已写好，别改）：
  - 所有块 id 唯一
  - 每块 text 非空，且除超长段外不超过 MAX_CHARS
  - 同一输入连续两次构建，id 序列完全一致
  - 页码都在有效范围内
"""
import hashlib
import re

from parse import PDF_PATH, assert_text_layer, clean_pages, find_repeated_lines, load_pages

DOC_ID = 'ali_java_manual'
MAX_CHARS = 500
OVERLAP = 80


def split_text(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """把一页文本切成若干段。

    切分优先级：
      1. 先按空行（段落）切
      2. 单段仍超长时，按句号 / 分号 / 问号 / 感叹号切
      3. 仍超长（罕见，如无标点的长表格行），按 max_chars 硬切
    相邻块之间保留 overlap 个字符的重复内容，避免答案正好卡在切口被切散。
    """
    # TODO 1: 实现切分
    #   建议先把段落合并成「不超过 max_chars 的块」，而不是一个段落一块—
    #   手册里段落普遍偏短，一段一块会导致块过碎、上下文不足。
    if not text.strip():
        return []
    return [seg.strip() for seg in re.split(r'\n\s*\n', text) if seg.strip()] or [text.strip()]


def make_chunk_id(doc_id, page, order):
    """生成稳定 ID。

    关键设计：只依赖 (doc_id, page, 页内序号)，不依赖正文。
      好处 —— 改切块参数只影响新增的块，老块 ID 不变，可以增量更新；
      代价 —— 如果某一页前面的块被合并，后面块的 order 会整体前移，ID 就变了。
    先用这个方案，等 W07 做重复导入时验证是否够用。
    """
    # TODO 2: 用 hashlib.sha1 对 f'{doc_id}:{page}:{order}' 取摘要前 12 位
    #   为什么不用 Python 内置 hash()：它带随机盐，进程重启后结果就变了
    #   所以要自己实现一个确定性函数，确保每次结果都一样
    return hashlib.sha1(f'{doc_id}:{page}:{order}'.encode()).hexdigest()[:12]


def build_chunks(pages, doc_id=DOC_ID):
    """把页列表转成块列表。"""
    # TODO 3: 逐页 split_text，枚举页内序号，组装 dict
    #   注意：空白页直接跳过，不要产出空块
    raise NotImplementedError('练习：实现 build_chunks')


if __name__ == '__main__':
    raw = load_pages(PDF_PATH)
    pages = clean_pages(raw, find_repeated_lines(raw))
    assert_text_layer(pages)

    chunks = build_chunks(pages)

    print(f'块数: {len(chunks)}')
    print(f'字符总数: {sum(len(c["text"]) for c in chunks)}')
    print('--- 示例块 ---')
    for c in chunks[:2]:
        print({k: (v[:60] + '...' if k == 'text' else v) for k, v in c.items()})

    # ===== 验收，不要改 =====
    ids = [c['id'] for c in chunks]
    assert len(ids) == len(set(ids)), '存在重复 id'
    assert all(c['text'].strip() for c in chunks), '存在空块'
    over = [c for c in chunks if len(c['text']) > MAX_CHARS * 2]
    assert not over, f'存在异常超长块 {len(over)} 个，检查切分逻辑'
    assert all(1 <= c['page'] <= len(pages) for c in chunks), '页码越界'

    again = build_chunks(pages)
    assert [c['id'] for c in again] == ids, '两次构建的 id 不一致，ID 不稳定'

    print('验收 PASS')