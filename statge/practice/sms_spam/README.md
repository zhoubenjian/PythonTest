# SMS 垃圾短信分类 · 骨架练习

目的：把 text_classification 项目里「看懂了」的链路，亲手「写出来」一遍。
换新数据集（UCI SMS Spam Collection，5572 条短文本，spam 占 13.4%），防止凭记忆抄旧代码。

## 数据

- 文件：`data/SMSSpamCollection`，UTF-8，每行 `ham<TAB>短信内容` 或 `spam<TAB>短信内容`，无表头
- 实测 5572 行：ham 4825 / spam 747（13.41%），无空值、无纯空白
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

- **练习 1 · `header=None` 和 `names` 各管一件事（11-16）**：`header=None` 表示「这个文件没有表头行」，`names=[...]` 表示「按我给的名字起列名」，两者不是一回事。只给 `names` 而不给 `header=None`，pandas 会把**第一行数据当成表头吞掉**，总行数少 1、列名也对不上。反过来，如果文件真有表头却写了 `header=None`，那行表头会被当成一条正常数据读进来（label 变成字符串 `'label'`），验收断言立刻失败。教训：**读数据前先确认文件有没有表头**，不能想当然。
- **练习 2 · `test_size` 是相对「传入的那个 DataFrame」算的（11-16）**：第二刀在 `rest_df`（85% 的数据）上划分，写 `test_size=0.15` 得到的是**全量的 4.5%**，不是 15%——结果 val 26% / test 5%。更值得记的是：**当时的验收断言只检查了「三份数据 spam 占比一致」，所以报 PASS 了，但结果是错的**。已给 split.py 补「比例 + 总量守恒」断言。教训：**断言决定了你能看见什么**，测试全绿不等于功能正确。
- **练习 3 · `f1_score` 必须喂离散标签（11-16）**：把 `predict_proba(...)[:, 1]` 的概率值直接传给 `f1_score` 会算出错误结果，因为指标内部做的是 `y_true == y_pred` 的逐元素比较，`0.837` 永远不等于 `1`。要用概率的话得自己设阈值转成 0/1：`(proba >= 0.5).astype(int)`——这也是阈值调优的入口。与练习 4 第二条（`predict_proba` 喂指标）同源，属同一盲区的两次表现。
- **练习 4 · accuracy 掩盖了少数类缺陷（11-16）**：test accuracy 0.9737 看着很漂亮，但混淆矩阵 `[[723, 1], [21, 91]]` 说明 22 条错分里 **21 条是把 spam 漏判成 ham**，spam recall 只有 **0.8125**。成因是 spam 仅占 13.4%，模型倾向于「不确定就猜 ham」。教训：不平衡数据必须看**每一类的 precision/recall 和混淆矩阵**，accuracy 单独出现时不能作为决策依据。附带红线：test 只跑一次，看到 spam recall 低也不能回 test 调阈值，只能在 val 上调、最后回 test 验一次。
- **练习 4 · `predict_proba` 不能直接喂评估指标（11-16）**：`model.predict_proba()` 返回 `(n, 2)` 概率矩阵，喂给 `accuracy_score` 会报 `ValueError: Classification metrics can't handle a mix of binary and continuous-multioutput targets`。判断标准：**要标签或指标用 `predict`，要概率/阈值/排序才用 `predict_proba`**。
