'''rebuilt API：FastAPI 入口。

运行方式（在 rebuilt 目录内；需先跑通 fit.py 生成 artifacts）：
    python -m uvicorn api.main:app

约束：不得 import 上级 src/ 的任何模块；运行目录即包路径，不用 sys.path 注入。
'''
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from api.model_service import ModelService, timed
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger('api')

service: ModelService


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    '''进程启动时加载一次产物（进程内单例）。

    产物缺失时这里抛 FileNotFoundError，uvicorn 启动失败并打印原因
    （早失败），而不是等第一个请求进来才报 500。
    '''
    global service
    service = ModelService()
    logger.info('service ready: n_features=%d classes=%s', service.n_features, service.classes)
    yield


app = FastAPI(title='rebuilt 文本分类 API', version='0.1.0', lifespan=lifespan)


@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    '''健康检查：顺带暴露模型特征维数与类别表，便于核对部署的是哪批产物。'''
    return HealthResponse(status='ok', n_features=service.n_features, classes=service.classes)


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    '''单条预测。'''
    try:
        result = timed(service, service.predict, req.text, req.top_k)
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(**result)


@app.post('/predict/batch', response_model=BatchPredictResponse)
def predict_batch_api(req: BatchPredictRequest) -> BatchPredictResponse:
    '''批量预测，results 顺序与 texts 一一对应。'''
    results = timed(service, service.predict_texts, req.texts, req.top_k)
    return BatchPredictResponse(count=len(results), results=[PredictResponse(**r) for r in results])
