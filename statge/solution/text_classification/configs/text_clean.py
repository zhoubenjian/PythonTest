'''20newsgroups 文本清洗：统一剥离邮件头 / footer / 引用

为什么需要这个模块
------------------
1. 原始帖子的邮件头里含 `Newsgroups: alt.atheism` 这类字段，直接等于标签。
   不剥离会让指标虚高（header 剥离实验实测：验证集 macro-F1 1.0000 -> 0.9744）。
2. sklearn 的 fetch_20newsgroups(remove=('headers','footers','quotes')) 会剥离，
   但项目在「sklearn 缓存未就绪」时会走 load_files 兜底读原始文件，默认不剥离。
3. 两条路径必须用同一套清洗规则，否则不同机器跑出的指标不可比。

规则（对齐 sklearn 的 remove 语义）
-----------------------------------
- headers：文本开头连续的 RFC822 邮件头区。要求连续 >= 2 个已知字段才判定为邮件头，
  避免把「正文首段恰好长得像 header」的帖子误删。
- footers：`--` 签名分隔行之后的内容。
- quotes ：以 `>` 开头的引用行。

用法
----
    from configs.text_clean import clean_text
    df['text'] = df['text'].map(clean_text)
'''
import re

# 20newsgroups 邮件头中出现频率最高的字段名（小写比较）
HEADER_FIELDS = (
    'newsgroups', 'path', 'from', 'subject', 'date', 'organization', 'lines',
    'distribution', 'xref', 'nntp-posting-host', 'nntp-posting-user', 'message-id',
    'references', 'reply-to', 'x-mailer', 'x-newsreader', 'keywords', 'summary',
    'originator', 'in-reply-to', 'followup-to', 'sender', 'approved',
    'content-type', 'content-transfer-encoding', 'mime-version', 'x-originating-ip',
)

HEADER_RE = re.compile(
    r'^(' + '|'.join(re.escape(f) for f in HEADER_FIELDS) + r')\s*:', re.I
)
QUOTE_RE = re.compile(r'^\s*>')
SIGNATURE_RE = re.compile(r'^--\s*$')


def _header_block_end(lines, max_scan=80):
    '''返回邮件头区域的结束行下标；若开头不像邮件头则返回 0（表示不剥离）。

    参数:
        lines:     按行切分后的文本
        max_scan:  最多扫描多少行，防止异常文本导致长循环
    返回:
        int 下标；0 表示「不剥离」
    '''
    hits = 0
    last = 0
    for i, line in enumerate(lines):
        if i > max_scan:  # 邮件头不会这么长，超了说明判断有误
            return 0
        last = i
        if line.strip() == '':
            return i + 1 if hits >= 2 else 0
        if HEADER_RE.match(line):
            hits += 1
        else:
            return 0  # 遇到非 header 行就判定开头不是邮件头区
    return last + 1 if hits >= 2 else 0


def strip_headers(text):
    '''去掉文本开头的邮件头区，保留正文。'''
    lines = text.split('\n')
    end = _header_block_end(lines)
    return '\n'.join(lines[end:]) if end else text


def strip_footer(text):
    '''去掉签名分隔行 `--` 之后的 footer（从后往前找，只取最后一处）。'''
    lines = text.split('\n')
    for i in range(len(lines) - 1, -1, -1):
        if SIGNATURE_RE.match(lines[i].strip()):
            return '\n'.join(lines[:i])
    return text


def strip_quotes(text):
    '''去掉以 `>` 开头的引用行。'''
    kept = [line for line in text.split('\n') if not QUOTE_RE.match(line)]
    return '\n'.join(kept)


def clean_text(text, headers=True, footers=True, quotes=True):
    '''按开关依次剥离邮件头 / footer / 引用，返回清洗后的文本。

    参数:
        text:    原始文本
        headers: 是否剥离邮件头
        footers: 是否剥离签名 footer
        quotes:  是否剥离引用行
    返回:
        str
    '''
    if not isinstance(text, str):
        return ''
    if headers:
        text = strip_headers(text)
    if footers:
        text = strip_footer(text)
    if quotes:
        text = strip_quotes(text)
    return text
