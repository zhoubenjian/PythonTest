"""练习 5/5（续）：API 测试

要求：至少 4 条测试，全部通过才算完成练习。
用 TestClient，不起真实服务：
    from fastapi.testclient import TestClient
    from api.main import app

4 条建议：
1. /health → 200 且 status == 'ok'
2. spam 风格文本（自己编一条：中奖/链接/免费领取之类）→ label == 'spam' 且 confidence ∈ (0.5, 1]
3. ham 风格文本（日常对话）→ label == 'ham' 且 confidence ∈ (0.5, 1]
4. 空文本 → 422 且外形 {code, message}，code == 'VALIDATION_ERROR'
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    """TODO: 实现"""
    raise NotImplementedError('练习：实现 test_health')


def test_spam_text():
    """TODO: 实现（自己编一条垃圾短信）"""
    raise NotImplementedError('练习：实现 test_spam_text')


def test_ham_text():
    """TODO: 实现（自己编一条正常短信）"""
    raise NotImplementedError('练习：实现 test_ham_text')


def test_empty_text_422():
    """TODO: 实现"""
    raise NotImplementedError('练习：实现 test_empty_text_422')
