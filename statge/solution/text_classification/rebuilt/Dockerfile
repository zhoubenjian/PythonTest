# rebuilt 文本分类 API 的最小运行镜像：sklearn + FastAPI，无训练依赖。
# 构建上下文必须是 rebuilt/（artifacts 与 predict.py 都要进镜像），即：
#   cd rebuilt
#   docker build -t text-class-api .
#   docker run --rm -p 8000:8000 text-class-api
# 构建时可换基础镜像源加速（国内直连 Docker Hub 常被 DNS 污染）：
#   --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# 构建时可换 pip 源加速：--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple

# scikit-learn 推理依赖 OpenMP 运行库，slim 基础镜像不带，缺了会在加载模型时报错
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api-requirements.txt .
RUN pip install --no-cache-dir -i "${PIP_INDEX_URL}" -r api-requirements.txt

# 只拷运行时必需三样：API 包、预测入口、产物；测试与文档不进镜像
COPY api/ ./api/
COPY predict.py ./
COPY artifacts/ ./artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
