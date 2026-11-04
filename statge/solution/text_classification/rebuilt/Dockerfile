# rebuilt 文本分类 API 的最小运行镜像：sklearn + FastAPI，无训练依赖。
# 构建上下文必须是 rebuilt/（artifacts 与 predict.py 都要进镜像），即：
#   cd rebuilt
#   docker build -t text-class-api .
#   docker run --rm -p 8000:8000 text-class-api
FROM python:3.11-slim

WORKDIR /app

# scikit-learn 推理依赖 OpenMP 运行库，slim 基础镜像不带，缺了会在加载模型时报错
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api-requirements.txt .
RUN pip install --no-cache-dir -r api-requirements.txt

# 只拷运行时必需三样：API 包、预测入口、产物；测试与文档不进镜像
COPY api/ ./api/
COPY predict.py ./
COPY artifacts/ ./artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
