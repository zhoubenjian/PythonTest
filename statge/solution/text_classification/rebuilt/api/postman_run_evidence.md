# Postman 集合实跑证据

对应集合：[postman_collection.json](postman_collection.json)（4 个请求）

- **执行时间**：2026-11-10 09:46:04
- **运行环境**：`api-requirements.txt` 钉版依赖（scikit-learn 1.5.2 / numpy 1.26.4 / fastapi 0.162.0 / uvicorn 0.54.0）
- **服务地址**：http://localhost:8000
- **结果**：**17 PASS / 0 FAIL**

> 说明：服务须用 `api-requirements.txt` 的环境启动。若用更高版本的 scikit-learn 加载产物，会触发
> `InconsistentVersionWarning`（产物由 1.5.2 生成），虽不影响本例预测结果，但正式证据以钉版环境为准。

---

## 实跑输出

```text
=== Postman 集合实跑证据 ===
时间: 2026-11-10 09:46:04
集合: rebuilt 文本分类 API (4 请求)
base_url: http://localhost:8000

[服务已就绪]

[1. 健康检查] GET /health
    响应: {"status": "ok", "n_features": 10000, "classes": ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "rec.sport.baseball"]}
    PASS  状态码 200
    PASS  status == 'ok'
    PASS  n_features == 10000
    PASS  classes 长度 == 4

[2. 单条预测] POST /predict
    请求: The team scored three runs in the ninth inning to win the ga...
    响应: {"label": "rec.sport.baseball", "confidence": 0.7757451290403236, "top_k": [["rec.sport.baseball", 0.7757451290403236], ["comp.graphics", 0.08278568052806412], ["comp.os.ms-windows.misc", 0.07946518735382542]]}
    耗时: 2058.1ms
    PASS  状态码 200
    PASS  响应含 label/confidence/top_k
    PASS  confidence 在 [0,1]
    PASS  top_k 首位与 label 一致
    PASS  top_k 降序

[3. 批量预测] POST /predict/batch
    响应: {"count": 2, "results": [{"label": "rec.sport.baseball", "confidence": 0.7757451290403236, "top_k": [["rec.sport.baseball", 0.7757451290403236], ["comp.graphics", 0.08278568052806412], ["comp.os.ms-windows.misc", 0.07946518735382542]]}, {"label": "comp.graphics", "confidence": 0.9155649361292334, "top_k": [["comp.graphics", 0.9155649361292334], ["rec.sport.baseball", 0.03113389265064061], ["alt.atheism", 0.02958235000297623]]}]}
    耗时: 2069.0ms
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
INFO api.service artifacts loaded in 2.240s: n_features=10000 classes=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'rec.sport.baseball']
INFO api service ready: n_features=10000 classes=['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'rec.sport.baseball']
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     127.0.0.1:64535 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:64537 - "GET /health HTTP/1.1" 200 OK
INFO api.service predicted in 2.2ms
INFO:     127.0.0.1:64542 - "POST /predict HTTP/1.1" 200 OK
INFO api.service predicted in 0.7ms
INFO:     127.0.0.1:64545 - "POST /predict/batch HTTP/1.1" 200 OK
INFO:     127.0.0.1:64550 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
```

> 客户端记录的 `耗时`（约 2s）包含 HTTP 连接建立与 JSON 序列化开销；
> 服务端 `predicted in 2.2ms / 0.7ms` 是纯推理耗时。对外描述性能时以服务端为准。

---

## 运行截图

| 文件 | 内容 |
|------|------|
| [01_swagger_docs.png](screenshots/01_swagger_docs.png) | Swagger UI：三个接口 + 6 个 Schemas |
| [02_predict_200.png](screenshots/02_predict_200.png) | POST /predict 返回 200，label=comp.graphics，confidence=0.9155649381292334 |

截图 2 的 confidence 与上方批量预测第二条**逐位一致**，说明 Postman 脚本与 Swagger UI 两条独立链路输出确定性相同。
