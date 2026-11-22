"""练习 2/3：流式接收

目标：看清 stream=True 时返回的到底是什么，以及为什么流式「不省 token」。

关键区别：
  - stream=False：一次性返回一个完整 JSON，等模型全部生成完
  - stream=True ：返回 text/event-stream（SSE），每次推一行 "data: {...}"，
                  最后一行是 "data: [DONE]"。整体不是合法 JSON，不能直接 resp.json()

每条 SSE 数据的结构：
  {"choices": [{"delta": {"content": "增量文本"}, "finish_reason": null}]}
  注意字段是 delta 而不是 message，content 是「本次增量」而不是全文，得自己累加。

验收（__main__ 里已写好，别改）：
  - 至少收到 2 个内容块（说明真的是分块来的）
  - 拼接后的全文非空
  - 首个内容块到达时间 < 整体结束时间
"""
import json
import time

import requests

from config import API_BASE, API_KEY, MODEL, TIMEOUT


def stream_chat(text):
    """流式请求，边收边打印，返回 (全文, 首块耗时秒, 内容块数)。"""

    url = f'{API_BASE}/chat/completions'

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }

    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': text}],
        'stream': True,  # 与练习 1 唯一的本质区别
    }

    started = time.perf_counter()
    first_chunk_at = None
    pieces = []

    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
        stream=True,
    )
    if not resp.ok:
        raise requests.HTTPError(
            f'HTTP {resp.status_code}: {resp.text[:500]}',
            response=resp,
        )

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith('data: '):
            continue
        body = raw_line[6:]
        if body == '[DONE]':
            break
        data = json.loads(body)
        delta = data['choices'][0].get('delta', {})
        content = delta.get('content')
        if content is None:
            continue
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter()
        pieces.append(content)
        print(content, end='', flush=True)

    elapsed_first = (first_chunk_at - started) if first_chunk_at else -1
    return ''.join(pieces), elapsed_first, len(pieces)


if __name__ == '__main__':
    full_text, first_chunk_at, n_chunks = stream_chat('用三句话解释什么是上下文窗口。')

    print('\n--- 汇总 ---')
    print(f'内容块数: {n_chunks}')
    print(f'首块耗时: {first_chunk_at:.3f}s')
    print(f'全文长度: {len(full_text)}')

    # ===== 验收，不要改 =====
    assert n_chunks >= 2, f'只收到 {n_chunks} 个内容块，可能没真正走流式'
    assert full_text.strip(), '拼接结果为空'
    assert 0 <= first_chunk_at < 999, '首块耗时未记录'
    print('验收 PASS')