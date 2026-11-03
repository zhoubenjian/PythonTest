'''rebuilt API：FastAPI 入口。

运行方式（在 rebuilt 目录内；需先跑通 fit.py 生成 artifacts）：
    python -m uvicorn api.main:app

约束：不得 import 上级 src/ 的任何模块；运行目录即包路径，不用 sys.path 注入。
'''
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_service import ModelService, timed
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
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


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    '''pydantic 校验失败（422）：裹成统一外形，只带前 3 条错误避免刷屏。'''
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code='VALIDATION_ERROR',
            message=f'输入校验失败: {exc.errors()[:3]}',
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def fallback_error_handler(_: Request, exc: Exception) -> JSONResponse:
    '''未知异常兜底（500）：完整堆栈进日志，调用方只见统一外形。'''
    logger.exception('unhandled error')
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(code='INTERNAL_ERROR', message='Internal server error').model_dump(),
    )


@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    '''健康检查：顺带暴露模型特征维数与类别表，便于核对部署的是哪批产物。'''
    return HealthResponse(status='ok', n_features=service.n_features, classes=service.classes)


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    '''单条预测。'''
    result = timed(service, service.predict, req.text, req.top_k)
    return PredictResponse(**result)


@app.post('/predict/batch', response_model=BatchPredictResponse)
def predict_batch_api(req: BatchPredictRequest) -> BatchPredictResponse:
    '''批量预测，results 顺序与 texts 一一对应。'''
    results = timed(service, service.predict_texts, req.texts, req.top_k)
    return BatchPredictResponse(count=len(results), results=[PredictResponse(**r) for r in results])
