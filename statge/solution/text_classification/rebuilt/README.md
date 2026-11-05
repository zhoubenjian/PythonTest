# rebuilt 文本分类：训练 → 预测 → API

独立重写版：不 import 原版 `src/` 的任何模块，只看行为与结论。与原版逐项差异见 `NOTES.md`。
数据集为 20 Newsgroups 的 4 个类别（alt.atheism / comp.graphics / comp.os.ms-windows.misc / rec.sport.baseball），
TF-IDF(10000 维) + LogisticRegression，验证集 macro-F1 = 0.9975。

## 目录

| 文件 | 职责 |
|---|---|
| `load.py` | 读 4 类原始数据（ISO-8859-1），返回文本与标签 |
| `split.py` | 8:1:1 分层切分，`random_state=42` |
| `fit.py` | 拟合 TF-IDF + 训练，产物写入 `artifacts/` |
| `predict.py` | 预测入口：加载产物，单条 / 批量预测 |
| `experiments.py` | W02 受控实验（去重 vs 保留、按组切分） |
| `error_analysis.py` | W02 错分归因分析（标注保留 + 分布汇总） |
| `artifacts/` | `vectorizer.pkl` + `model.pkl`，训练产物 |
| `api/` | FastAPI 服务（`main.py` / `schemas.py` / `model_service.py` / `test_api.py`） |
| `Dockerfile` / `api-requirements.txt` / `.dockerignore` | 最小运行镜像 |
| `api/postman_collection.json` | 四请求验证集合 |

## 环境

三端（Windows / Ubuntu / Mac）统一用 conda 环境 `ai_app`（Python 3.11），钉版见上级 `environment.yml`：

```bash
conda activate ai_app
```

API 依赖（本地开发需要；容器内由 `api-requirements.txt` 提供）：

```bash
pip install fastapi==0.162.0 uvicorn==0.54.0 httpx==0.28.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 1. 训练

```bash
cd rebuilt
python fit.py
```

产出 `artifacts/vectorizer.pkl`、`artifacts/model.pkl`，并打印验证集指标。
`RANDOM_STATE=42` 固定，重跑结果可复现（同一环境重训产物字节一致）。

## 2. 命令行预测

```bash
cd rebuilt
python predict.py
```

## 3. HTTP API

**必须在 `rebuilt/` 目录下启动**：`api` 是本地包、`predict.py` 在上一层，uvicorn 会把启动目录加入 `sys.path`。

```bash
cd rebuilt
python -m uvicorn api.main:app
```

启动成功会打印 `service ready: n_features=10000 classes=[...]`；交互式文档在 http://127.0.0.1:8000/docs 。

### 接口

| 方法 | 路径 | 说明 | 请求体 |
|---|---|---|---|
| GET | `/health` | 健康检查，返回模型特征维数与类别表 | 无 |
| POST | `/predict` | 单条预测，返回 top-k 类别与概率 | `{"text": "...", "top_k": 3}` |
| POST | `/predict/batch` | 批量预测，`results` 顺序与 `texts` 一一对应 | `{"texts": ["...", "..."], "top_k": 3}` |

约束：`text` 不能为空；`top_k` 取值 1..10；`texts` 长度 1..100。`top_k` 超过类别数时由服务层收敛到 4。

### curl 验证

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The team scored three runs in the ninth inning to win the game.","top_k":3}'

curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Computer graphics rendering uses polygon meshes and shaders.","The team scored three runs in the ninth inning."], "top_k": 3}'

# 非法输入：期望 422 与统一错误外形
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"text":""}'
```

示例响应（单条）：

```json
{
  "label": "rec.sport.baseball",
  "confidence": 0.7757,
  "top_k": [["rec.sport.baseball", 0.7757], ["comp.graphics", 0.1203], ["comp.os.ms-windows.misc", 0.0811], ["alt.atheism", 0.0229]]
}
```

Windows PowerShell 等价写法（中文文本建议用 curl 或显式 UTF-8，PowerShell 默认编码会出问题）：

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post `
  -ContentType 'application/json' -Body '{"text":"The team scored three runs in the ninth inning."}'
```

### 错误处理

所有错误返回同一外形 `{"code": "...", "message": "..."}`，调用方只需写一次错误处理。

| 场景 | 状态码 | code |
|---|---|---|
| 请求体校验失败（空文本 / top_k 越界 / texts 不是数组 / 超过 100 条） | 422 | `VALIDATION_ERROR` |
| 未预期异常（兜底，堆栈只进日志） | 500 | `INTERNAL_ERROR` |
| 产物缺失 | 进程启动失败 | —（uvicorn 打印 `FileNotFoundError`，不进入服务状态） |

产物缺失采用**早失败**：`lifespan` 启动时加载产物，失败就不启动，不会让第一个请求去承担 500。
因此没有 `SERVICE_UNAVAILABLE` 这类 503 —— 服务不可用时根本起不来，写 503 分支属于死代码。

### Postman

导入 `api/postman_collection.json`，四个请求：健康检查 / 单条预测 / 批量预测 / 非法输入（期望 422），
每个请求带断言脚本。集合变量 `base_url` 默认 `http://localhost:8000`，换地址只改变量。

## 4. 测试

```bash
cd rebuilt
python -m pytest api/test_api.py -v
```

7 条用例，用真实 `artifacts/`（不 mock 模型），只断言结构与不变量：类别表、概率范围、top_k 降序、
批量与单条结果一致、非法输入 422、错误外形统一。`artifacts/` 缺失时整文件 skip 而不是失败。

## 5. Docker

镜像只装 7 个运行时依赖（`api-requirements.txt` 钉版），不含 pandas / matplotlib / pytest 等训练与开发依赖。

```bash
cd rebuilt
docker build -t text-class-api .
docker run --rm -d -p 8000:8000 text-class-api
curl -s http://localhost:8000/health
docker logs -f $(docker ps -q --filter ancestor=text-class-api)
```

国内网络（Docker Hub 被 DNS 污染时）换源构建，默认值保持官方源：

```bash
docker build \
  --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t text-class-api .
```

镜像内含 `artifacts/`，因此产物换了要重新构建（不挂卷，避免镜像内产物与外部不一致）。

## 6. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `ARTIFACTS_DIR` | `rebuilt/artifacts` | 产物目录，多套产物切换或挂载时用 |
| `LOG_LEVEL` | `INFO` | 日志级别；非法值启动即报错（早失败） |

## 设计取舍

不在此展开，统一见 `NOTES.md` 的差异表：`sys.path` 注入、LabelEncoder、产物隔离、选型口径、
错误处理的一致性原则，每一条都写了「原版怎么做 / 我怎么做 / 为什么」。
