"""练习 5/5：FastAPI 服务化

前置：evaluate.py 已跑完，artifacts/ 下有 model.pkl 和 vectorizer.pkl。

接口约定：
- GET  /health  → 200 {"status": "ok", "model": "sms-spam"}
- POST /predict 请求体 {"text": "<短信内容>"}
       → 200 {"label": "spam"|"ham", "confidence": <0~1 小数>}
       → 空文本/纯空白 → 422 {"code": "VALIDATION_ERROR", "message": "..."}
- 统一异常外形 {code, message}；confidence 是「预测为该 label 的概率」

启动（8100 端口，避开老服务）：uvicorn api.main:app --port 8100
测试：在上级目录运行 pytest api/test_api.py

注意：
- 模型加载一次、全局复用，不要每个请求都读盘
- predict_proba 的列顺序由 model.classes_ 决定，别想当然
"""
import os

from fastapi import FastAPI

app = FastAPI(title='sms-spam API', version='0.1.0')

# TODO: 启动时加载 model.pkl / vectorizer.pkl（怎么拿相对路径才稳？）


@app.get('/health')
def health():
    """TODO: 返回约定外形"""
    raise NotImplementedError('练习：实现 health')


@app.post('/predict')
def predict(body):
    """TODO:
      1. 定义请求体 Pydantic 模型（text 字段，长度约束在哪做？）
      2. 空文本 → 422 VALIDATION_ERROR（外形 {code, message}）
      3. 向量化 → predict_proba → label + confidence
      4. 返回 {"label": ..., "confidence": ...}
    """
    raise NotImplementedError('练习：实现 predict')
