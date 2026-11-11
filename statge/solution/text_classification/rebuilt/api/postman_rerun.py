"""按 postman_collection.json 的 4 个请求重跑一遍断言，输出实跑证据。

用法：服务起在 8000 后，python postman_rerun.py
"""
import json
import urllib.request
import urllib.error
import time

BASE = 'http://localhost:8000'
passed = 0
failed = 0


def check(name, ok):
    global passed, failed
    print(f'    {"PASS" if ok else "FAIL"}  {name}')
    if ok:
        passed += 1
    else:
        failed += 1


def call(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={'Content-Type': 'application/json'},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode()), (time.time() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode()), (time.time() - t0) * 1000


print('=== Postman 集合实跑证据（清洗后重跑）===')
print(f'时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
print('集合: rebuilt 文本分类 API (4 请求)')
print(f'base_url: {BASE}')
print()

# 1. 健康检查
print('[1. 健康检查] GET /health')
code, body, ms = call('GET', '/health')
print(f'    响应: {json.dumps(body, ensure_ascii=False)}')
check('状态码 200', code == 200)
check("status == 'ok'", body.get('status') == 'ok')
check('n_features == 10000', body.get('n_features') == 10000)
check('classes 长度 == 4', len(body.get('classes', [])) == 4)
print()

# 2. 单条预测
print('[2. 单条预测] POST /predict')
text = 'The team scored three runs in the ninth inning to win the game.'
code, body, ms = call('POST', '/predict', {'text': text, 'top_k': 3})
print(f'    请求: {text[:60]}...')
print(f'    响应: {json.dumps(body, ensure_ascii=False)}')
print(f'    耗时: {ms:.1f}ms')
check('状态码 200', code == 200)
check('响应含 label/confidence/top_k', all(k in body for k in ('label', 'confidence', 'top_k')))
check('confidence 在 [0,1]', 0 <= body.get('confidence', -1) <= 1)
tk = body.get('top_k', [])
check('top_k 首位与 label 一致', bool(tk) and tk[0][0] == body.get('label'))
check('top_k 降序', all(tk[i][1] >= tk[i + 1][1] for i in range(len(tk) - 1)))
print()

# 3. 批量预测
print('[3. 批量预测] POST /predict/batch')
texts = [
    'The team scored three runs in the ninth inning to win the game.',
    'Computer graphics rendering uses polygon meshes and shaders.',
]
code, body, ms = call('POST', '/predict/batch', {'texts': texts, 'top_k': 3})
print(f'    响应: {json.dumps(body, ensure_ascii=False)}')
print(f'    耗时: {ms:.1f}ms')
check('状态码 200', code == 200)
check('count == 2', body.get('count') == 2)
results = body.get('results', [])
check('results 长度 == 2', len(results) == 2)
ok = bool(results) and all(
    all(k in r for k in ('label', 'confidence', 'top_k')) and r['top_k'][0][0] == r['label']
    for r in results[:1]
)
check('results[0] 结构完整且 top_k 首位==label', ok)
ok = len(results) > 1 and all(
    all(k in r for k in ('label', 'confidence', 'top_k')) and r['top_k'][0][0] == r['label']
    for r in results[1:]
)
check('results[1] 结构完整且 top_k 首位==label', ok)
print()

# 4. 非法输入
print('[4. 非法输入] POST /predict 空文本')
code, body, ms = call('POST', '/predict', {'text': '', 'top_k': 3})
print(f'    响应: {json.dumps(body, ensure_ascii=False)[:260]}')
check('状态码 422', code == 422)
check('错误外形 == {code, message}', set(body.keys()) == {'code', 'message'})
check("code == VALIDATION_ERROR", body.get('code') == 'VALIDATION_ERROR')
print()

print(f'=== 汇总: {passed} PASS / {failed} FAIL ===')
