# Postman 集合实跑证据

对应集合：[postman_collection.json](postman_collection.json)（4 个请求）
复跑脚本：[postman_rerun.py](postman_rerun.py)（与服务同环境，直接跑即可复现）

- **执行时间**：2026-11-11 11:16:48（数据清洗后重跑）
- **运行环境**：`api-requirements.txt` 钉版依赖（scikit-learn 1.5.2 / numpy 1.26.4 / pandas 2.2.3 / fastapi / uvicorn）
- **服务地址**：http://localhost:8000
- **结果**：**17 PASS / 0 FAIL**

> 重要：本文件在 2026-11-11 数据口径修正后重新生成。
> 原记录（2026-11-10）的置信度数值（0.7757 / 0.9156）来自**未剥离邮件头**的数据，
> 存在标签泄漏，已作废。当前数值基于 `configs/text_clean.py` 统一清洗后的数据。

---

## 实跑输出

```text
=== Postman 集合实跑证据（清洗后重跑）===
时间: 2026-11-11 11:16:48
集合: rebuilt 文本分类 API (4 请求)
base_url: http://localhost:8000

[1. 健康检查] GET /health
    响应: {"status": "ok", "n_features": 10000, "classes": ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "rec.sport.baseball"]}
    PASS  状态码 200
    PASS  status == 'ok'
    PASS  n_features == 10000
    PASS  classes 长度 == 4

[2. 单条预测] POST /predict
    请求: The team scored three runs in the ninth inning to win the ga...
    响应: {"label": "rec.sport.baseball", "confidence": 0.8917835358288766, "top_k": [["rec.sport.baseball", 0.8917835358288766], ["comp.os.ms-windows.misc", 0.05027751012545256], ["comp.graphics", 0.03416951373422382]]}
    耗时: 2047.2ms
    PASS  状态码 200
    PASS  响应含 label/confidence/top_k
    PASS  confidence 在 [0,1]
    PASS  top_k 首位与 label 一致
    PASS  top_k 降序

[3. 批量预测] POST /predict/batch
    响应: {"count": 2, "results": [{"label": "rec.sport.baseball", "confidence": 0.8917835358288766, "top_k": [["rec.sport.baseball", 0.8917835358288766], ["comp.os.ms-windows.misc", 0.05027751012545256], ["comp.graphics", 0.03416951373422382]]}, {"label": "comp.graphics", "confidence": 0.9437804495710632, "top_k": [["comp.graphics", 0.9437804495710632], ["comp.os.ms-windows.misc", 0.024122484472750814], ["rec.sport.baseball", 0.01644310079985114]]}]}
    耗时: 2051.2ms
    PASS  状态码 200
    PASS  count == 2
    PASS  results 长度 == 2
    PASS  results[0] 结构完整且 top_k 首位==label
    PASS  results[1] 结构完整且 top_k 首位==label

[4. 非法输入] POST /predict 空文本
    响应: {"code": "VALIDATION_ERROR", "message": "输入校验失败: [{'type': 'string_too_short', 'loc': ('body', 'text'), 'msg': 'String should have at least 1 character', 'input': '', 'ctx': {'min_length': 1}}]"}
    PASS  状态码 422
    PASS  错误外形 == {code, message}
    PASS  code == VALIDATION_ERROR

=== 汇总: 17 PASS / 0 FAIL ===
```

---

## 服务端日志对照

```text
INFO api.service artifacts loaded in 3.890s: n_features=10000 classes=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'rec.sport.baseball']
INFO api service ready: n_features=10000 classes=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'rec.sport.baseball']
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     127.0.0.1:56291 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:56343 - "GET /health HTTP/1.1" 200 OK
INFO api.service predicted in 3.6ms
INFO:     127.0.0.1:56347 - "POST /predict HTTP/1.1" 200 OK
INFO api.service predicted in 1.4ms
INFO:     127.0.0.1:56351 - "POST /predict/batch HTTP/1.1" 200 OK
INFO:     127.0.0.1:56370 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
```

> 客户端记录的 `耗时`（约 2s）包含 HTTP 连接建立与 JSON 序列化开销；
> 服务端 `predicted in 3.6ms / 1.4ms` 是纯推理耗时。对外描述性能时以服务端为准。

---

## 运行截图

| 文件 | 内容 | 状态 |
|------|------|------|
| [01_swagger_docs.png](screenshots/01_swagger_docs.png) | Swagger UI：三个接口 + 6 个 Schemas | 有效（不含数值） |
| [02_predict_200.png](screenshots/02_predict_200.png) | POST /predict 返回 200 | **需重截**：图中 confidence 为旧模型值 |

截图 02 应重截为 `comp.graphics / 0.9437804495710632`（批量结果第二条），
与上表数值逐位一致即为有效证据。
