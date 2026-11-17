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
import joblib

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title='sms-spam API', version='0.1.0')

# TODO: 启动时加载 model.pkl / vectorizer.pkl（怎么拿相对路径才稳？）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'artifacts/model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'artifacts/vectorizer.pkl'))


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

    @field_validator('text')
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError('text 不能为空')
        return v


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    msg = exc.errors()[0].get('msg', '参数校验失败')
    return JSONResponse(
        status_code=422,
        content={'code': 'VALIDATION_ERROR', 'message': msg},
    )


@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'sms-spam'}


@app.post('/predict')
def predict(body: PredictRequest):
    # 只transform，不fit
    X = vectorizer.transform([body.text])

    # 拿到各类概率
    proba = model.predict_proba(X)[0]

    # 取最大概率索引
    idx = proba.argmax()

    # 用 classes_ 映射回标签
    label = model.classes_[idx]

    return {'label': label, 'confidence': float(proba[idx])}
