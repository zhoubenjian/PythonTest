# SMS 垃圾短信分类 · 骨架练习

目的：把 text_classification 项目里「看懂了」的链路，亲手「写出来」一遍。
换新数据集（UCI SMS Spam Collection，5574 条短文本，spam 占 13.4%），防止凭记忆抄旧代码。

## 数据

- 文件：`data/SMSSpamCollection`，UTF-8，每行 `ham<TAB>短信内容` 或 `spam<TAB>短信内容`，无表头
- 类别不平衡（spam 约 13.4%）——这会逼你重新思考该用什么指标

## 规则（重要，先读完再动手）

1. **按顺序做**：load → split → fit → evaluate → api，每个文件跑到验收 PASS 才进下一个
2. **先自己写**，写完运行 `python xxx.py` 看验收输出
3. **卡住超过 20 分钟** → 才能看 [hints.md](hints.md)，且**只看当前阶段的提示**（阶段 1 → 2 → 3，尽量别看到阶段 3）
4. 每一步的踩坑（报了什么错、怎么查的、怎么解的）**记到本文件末尾的「踩坑记录」**——这是面试素材，比跑通更值钱
5. 做完不急着提交：先自己对着每一步说出「为什么这么做」，说不出的地方就是下一个要补的洞
6. 明天是投递日，**投递优先**，练习穿插着做

## 步骤与验收标准

| # | 文件 | 任务 | 验收 |
|---|------|------|------|
| 1 | [load.py](load.py) | 读 TSV 成 DataFrame | 列名/取值/无空值断言通过 |
| 2 | [split.py](split.py) | 分层划分 70/15/15 | 三份 spam 占比差 < 1 个百分点 |
| 3 | [fit.py](fit.py) | TF-IDF + 两模型对比选型 | 最佳 val macro-F1 ≥ 0.85 |
| 4 | [evaluate.py](evaluate.py) | 测试集终评（只跑一次） | test F1 与 val F1 差 < 0.05 |
| 5 | [api/main.py](api/main.py) | FastAPI 两个端点 + 统一错误 | 手工调用符合约定 |
|   | [api/test_api.py](api/test_api.py) | ≥ 4 条测试 | pytest 全 PASSED |

启动 API 用 **8100 端口**（避开 8000 的老服务）：`uvicorn api.main:app --port 8100`

## 与面试考点的对应

| 练习步骤 | 面试问题 |
|----------|----------|
| split 分层 | 为什么必须 stratify？不平衡数据随机划分会怎样？ |
| fit 选型 | 为什么用 macro-F1 不用 accuracy？99% 准确率可能很差吗？ |
| fit | TF-IDF 只在训练集上 fit，为什么？在全量上 fit 是什么泄漏？ |
| evaluate | 测试集为什么只能跑一次？跑多了会发生什么？ |
| api | confidence 怎么算？predict_proba 的列顺序由什么决定？ |

## 踩坑记录（每步做完就补一行）

- （待填）
