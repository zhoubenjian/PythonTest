"""练习 5/5（续）：API 测试

要求：至少 4 条测试，全部通过才算完成练习。
用 TestClient，不起真实服务：
    from fastapi.testclient import TestClient
    from api.main import app

4 条建议：
1. /health => 200 且 status == 'ok'
2. spam 风格文本（自己编一条：中奖/链接/免费领取之类）=> label == 'spam' 且 confidence ∈ (0.5, 1]
3. ham 风格文本（日常对话）=> label == 'ham' 且 confidence ∈ (0.5, 1]
4. 空文本 => 422 且外形 {code, message}，code == 'VALIDATION_ERROR'
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert data['model'] == 'sms-spam'


def test_spam_text():
    # 1. 模拟明显垃圾短信
    spam_text = '恭喜你抽中iPhone18！点击链接即刻领取：http://fraud-link.com/win 回复1234到876543确认'

    # 2. 发送POST请求
    response = client.post('/predict', json={'text': spam_text})

    # 3. 断言状态码
    assert response.status_code == 200

    # 4. 断言响应内容
    data = response.json()
    assert data['label'] == 'spam'
    conf = data['confidence']
    assert 0.5 < conf <= 1.0


def test_ham_text():
    # 1. 模拟正常短信
    ham_text = '你好，我是Alex，我想知道我的订单状态。'

    # 2. 发送POST请求
    response = client.post('/predict', json={'text': ham_text})

    # 3. 断言状态码
    assert response.status_code == 200

    # 4. 断言响应内容
    data = response.json()
    assert data['label'] == 'ham'
    conf = data['confidence']
    assert 0.5 < conf <= 1.0


def test_empty_text_422():
    '''
    空字符串触发 min_length=1
    :return:
    '''

    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 422
    data = response.json()
    assert data['code'] == 'VALIDATION_ERROR'
    assert 'at least 1 character' in data['message']


def test_blank_text_422():
    '''
    纯空白触发自定义 not_blank validator    
    :return:
    '''

    response = client.post('/predict', json={'text': '   '})
    assert response.status_code == 422
    data = response.json()
    assert data['code'] == 'VALIDATION_ERROR'
    assert '不能为空' in data['message']
