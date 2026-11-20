"""练习 1/3：非流式调用，看清一次请求 / 响应的全貌

目标（不查 SDK 文档，只看 HTTP）：
  1. headers 里到底要带什么
  2. payload 的 messages 是什么结构
  3. 回答正文在响应的哪个字段，token 用量在哪个字段

接口信息：
  方法/路径：POST {API_BASE}/chat/completions
  鉴权：Authorization: Bearer <API_KEY>
  响应：{"choices": [{"message": {"role", "content"}}],
         "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}}

验收（__main__ 里已写好，别改）：
  - content 非空
  - usage 三个字段都存在且 > 0
  - prompt_tokens + completion_tokens == total_tokens
"""
import requests

from config import API_BASE, API_KEY, MODEL, TIMEOUT


def call_once(text):
    """发一次非流式请求，返回 (content, usage)。"""

    url = f'{API_BASE}/chat/completions'

    # 1: 组装 headers
    #   至少要有 Authorization（Bearer 开头）和 Content-Type: application/json
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # 2: 组装 payload
    #   model 用 MODEL
    #   messages 是 [{'role': 'user', 'content': text}]
    #   显式写下 stream: False，让「非流式」这件事可见
    payload = {
        'model': MODEL,
        'messages': [{
            'role': 'user',
            'content': text
        }],
        'stream': False
    }

    # 3: 发 POST
    #   用 json= 而不是 data=，requests 会自己序列化并带上正确的 Content-Type
    #   一定要传 timeout，线上没超时的请求就是事故
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=TIMEOUT
    )

    # 4: 先判 HTTP 状态，再把响应体转 dict
    #   非 2xx 时把状态码和响应文本一起抛出来，方便定位
    if not resp.ok:
        raise requests.HTTPError(
            f'HTTP {resp.status_code}: {resp.text[:500]}',
            response=resp,
        )
    data = resp.json()

    # 5: 从 data 里取回答正文和用量
    #   content 在 choices[0] -> message -> content
    #   usage 直接就是 data 里那一段
    content = data['choices'][0]['message']['content']
    usage = data['usage']

    return content, usage


if __name__ == '__main__':
    content, usage = call_once('用一句话说明什么是 token。')

    print('--- 回答 ---')
    print(content)
    print('--- 用量 ---')
    print(usage)

    # ===== 验收，不要改 =====
    assert content.strip(), '回答为空'
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        assert key in usage and usage[key] > 0, f'usage 缺字段或非正数：{key}'
    assert usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens'], 'token 不守恒'
    print('验收 PASS')