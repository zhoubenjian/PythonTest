'''rebuilt API：请求/响应模型（pydantic）。

校验放边界上：进来的 JSON 在这里被拦掉非法输入（空文本、越界 top_k），
内部的 model_service 只处理已合法化的数据。
'''
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    '''单条预测请求。top_k 限制在 1..10，服务端还会按类别数再收敛一次。'''
    text: str = Field(min_length=1, description='待分类文本')
    top_k: int = Field(default=3, ge=1, le=10, description='返回概率最高的前 k 个类别')


class PredictResponse(BaseModel):
    '''单条预测响应。label/confidence 是纯 Python 类型——
    numpy 标量过不了 JSON 序列化，类型转换在 service 层完成。'''
    label: str
    confidence: float
    top_k: list[tuple[str, float]]


class BatchPredictRequest(BaseModel):
    '''批量预测请求。texts 上限 100：API 是交付演示场景，不做无上界的批量。'''
    texts: list[str] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=3, ge=1, le=10)


class BatchPredictResponse(BaseModel):
    '''批量预测响应。results 顺序与请求 texts 一一对应。'''
    count: int
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    '''健康检查响应。n_features/classes 来自加载后的产物，便于核对部署的模型。'''
    status: str
    n_features: int
    classes: list[str]


class ErrorResponse(BaseModel):
    '''统一错误响应体：所有错误场景返回同一外形，调用方的错误处理只写一次。

    code 是机器可读的错误码（如 VALIDATION_ERROR / INTERNAL_ERROR），
    供调用方程序分支；HTTP 状态码仍负责粗分类。
    '''
    code: str = Field(examples=['VALIDATION_ERROR'], description='机器可读错误码')
    message: str = Field(description='人类可读说明，只描述问题不泄漏内部细节')
