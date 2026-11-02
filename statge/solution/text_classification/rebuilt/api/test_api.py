'''rebuilt API：pytest 接口测试（FastAPI TestClient，不真起端口）。

运行方式（在 rebuilt 目录内，需先跑通 fit.py 生成 artifacts）：
    python -m pytest api/test_api.py -v

测试对齐三条原则：
1. 不 mock 模型——直接用真实 artifacts，测的是交付物本身；
2. artifacts 缺失时整文件 skip，而不是失败（同原版 tests 的处理）；
3. 只断言结构与不变量（类别表、概率范围、批量与单条一致），
   不断言具体类别值——模型换数据重训后测试不应失效。
'''
import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

_API_DIR = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS = [
    os.path.join(_API_DIR, '..', 'artifacts', f) for f in ('vectorizer.pkl', 'model.pkl')
]

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in _ARTIFACTS),
    reason='artifacts 缺失，先运行 fit.py',
)


@pytest.fixture(scope='module')
def client() -> TestClient:
    '''module 级 fixture：产物只加载一次，全部用例共用同一个进程内服务。'''
    with TestClient(app) as c:
        yield c


SAMPLES = [
    'The team scored three runs in the ninth inning to win the game.',
    'Computer graphics rendering uses polygon meshes and shaders.',
    'Quantum error correction requires fault-tolerant logical qubits.',
]


def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] == 'ok'
    assert body['n_features'] == 10000
    assert len(body['classes']) == 4


def test_predict_single(client):
    resp = client.post('/predict', json={'text': SAMPLES[0]})
    assert resp.status_code == 200
    body = resp.json()
    assert body['label'] in client.get('/health').json()['classes']
    assert 0.0 <= body['confidence'] <= 1.0
    # top_k 按概率降序，且第一名就是 label 本身
    top = body['top_k']
    assert top[0][0] == body['label']
    assert top[0][1] == body['confidence']
    assert all(top[i][1] >= top[i + 1][1] for i in range(len(top) - 1))


def test_predict_top_k_clamped(client):
    '''schema 拦 top_k>10；类别数只有 4 时 service 再收敛到 4——两层各管各的。'''
    resp = client.post('/predict', json={'text': SAMPLES[1], 'top_k': 10})
    assert resp.status_code == 200
    assert len(resp.json()['top_k']) == 4


def test_batch_matches_single(client):
    '''批量与单条对同一文本给出相同结果——顺序对应的关键不变量。'''
    single = client.post('/predict', json={'text': SAMPLES[0]}).json()
    resp = client.post('/predict/batch', json={'texts': SAMPLES[:2]})
    assert resp.status_code == 200
    body = resp.json()
    assert body['count'] == 2
    assert len(body['results']) == 2
    assert body['results'][0]['label'] == single['label']
    assert body['results'][0]['confidence'] == single['confidence']


def test_batch_rejects_single_string(client):
    '''单个 str 会被逐字切分——接口层直接 422 拦掉（与 predict_batch 的 TypeError 契约一致）。'''
    resp = client.post('/predict/batch', json={'texts': 'not a list'})
    assert resp.status_code == 422


def test_predict_rejects_empty_text(client):
    resp = client.post('/predict', json={'text': ''})
    assert resp.status_code == 422
