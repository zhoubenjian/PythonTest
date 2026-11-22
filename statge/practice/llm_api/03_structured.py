"""练习 3/3：结构化输出 + 错误处理

目标：让模型稳定返回能直接吃的 JSON，并把「网络/鉴权/模型不听话」这三类失败挡住。

两个关键手段：
  1. response_format={'type': 'json_object'}：告诉模型必须输出 JSON。
     注意：DeepSeek 要求 prompt 里必须出现 "json" 字样，否则会报错。
  2. 兜底解析：即便要求了 JSON，模型仍可能包 ```json 围栏或前后加解释，
     所以解析函数要能清理，失败时抛 ValueError 而不是静默返回 None。

接口契约（自己实现时对照）：
  正常返回 dict，至少含 summary(str) 和 tags(list[str])
"""
import json
import time

import requests

from config import API_BASE, API_KEY, MODEL, TIMEOUT

SCHEMA_HINT = (
    '请只输出 JSON，不要任何额外文字。格式：'
    '{"summary": "一句话摘要", "tags": ["标签1", "标签2"]}'
)


def parse_json(text):
    """把模型返回的文本解析成 dict。

    要处理三种情况：
      1. 纯 JSON          -> 直接 loads
      2. ```json ... ``` 围栏 -> 先剥掉围栏再 loads
      3. 根本不是 JSON     -> 抛 ValueError，带上原始片段方便排查
    """
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        body_lines = []
        started = False
        for line in lines:
            if not started:
                if line.strip().startswith('```'):
                    started = True
                continue
            if line.strip() == '```':
                break
            body_lines.append(line)
        text = '\n'.join(body_lines).strip()

    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f'未找到 JSON 对象边界：{text[:100]!r}')
    text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f'JSON 解析失败：{e.msg}，片段：{text[:100]!r}') from e


def call_json(text, retries=2):
    """调用模型拿 JSON，对超时和 429 做有限重试。"""

    url = f'{API_BASE}/chat/completions'
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': '你是严格输出 JSON 的助手。'},
            {'role': 'user', 'content': f'{text}\n\n{SCHEMA_HINT}'},
        ],
        'response_format': {'type': 'json_object'},
        'stream': False,
    }

    last_err = None
    for i in range(retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
        except requests.Timeout as e:
            last_err = e
            if i < retries:
                time.sleep(0.5 * 2 ** i)
                continue
            raise

        if resp.status_code in (401, 403):
            raise requests.HTTPError(
                f'HTTP {resp.status_code}：鉴权失败，不重试',
                response=resp,
            )

        if resp.status_code == 429:
            last_err = requests.HTTPError(
                f'HTTP 429：限流',
                response=resp,
            )
            if i < retries:
                time.sleep(0.5 * 2 ** i)
                continue
            raise last_err

        if not resp.ok:
            raise requests.HTTPError(
                f'HTTP {resp.status_code}: {resp.text[:500]}',
                response=resp,
            )

        content = resp.json()['choices'][0]['message']['content']
        return parse_json(content)

    raise last_err


if __name__ == '__main__':
    # ===== 验收 1：坏 JSON 必须抛 ValueError（离线可测，不消耗额度）=====
    for bad in ['这不是 json', '', '{"a": 1']:
        try:
            parse_json(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'应当抛 ValueError：{bad!r}')

    # ===== 验收 2：markdown 围栏能被剥掉（离线可测）=====
    fenced = '```json\n{"summary": "测试", "tags": ["a"]}\n```'
    obj = parse_json(fenced)
    assert isinstance(obj, dict) and obj.get('summary') == '测试', '围栏未正确剥离'

    # ===== 验收 3：真实调用返回符合契约（需要网络）=====
    result = call_json('总结这句话：今天学习了如何调用大模型 API。')
    assert isinstance(result, dict), '返回值不是 dict'
    assert isinstance(result.get('summary'), str) and result['summary'].strip(), 'summary 缺失'
    assert isinstance(result.get('tags'), list) and result['tags'], 'tags 缺失或为空'

    print('--- 抽取结果 ---')
    print(result)
    print('验收 PASS')