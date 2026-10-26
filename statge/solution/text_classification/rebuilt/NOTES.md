# rebuilt 与 原实现 的差异对照

独立重写完成后再填这张表。每一行必须是"我确实这么做了"的差异，不是猜测。
填不出来的格子写"无"——"无差异"本身就是结论。

## 差异表

| 环节 | 原版做法（src/ + configs/） | 我的做法（rebuilt/） | 理由 |
|---|---|---|---|
| 数据加载 | `load_files` 读 `local_datas/20_newsgroups`，ISO-8859-1，固定 4 类 | 同源同参数（`load.py`），行为一致 | 数据源和编码没有分歧的理由，保持一致以便 F1 可直接对比 |
| 数据路径解析 | `configs/config.py` 集中定义全部常量 | 模块级常量就近放（`DATA_ROOT`、`ARTIFACT_DIR`），无集中配置文件 | 只有 5 个文件时集中配置是过度设计；每个文件自包含，单独可读。文件多了再收回集中式 |
| 模块导入方式 | `sys.path.insert(...)` 注入路径 | 无注入，靠"在 rebuilt 目录内运行" | sys.path.insert 是隐式全局副作用：换机器、换 IDE、换启动方式都可能断；小项目用运行目录更可移植 |
| 训练/验证/测试划分 | 两次 `train_test_split`，8:1:1 分层，`random_state=42` | 同为两次切分分层（`split.py`），比例与种子一致 | 划分一致才能把 F1 差异归因到实现而非数据；实测差 1 条，已归因为 cross-post（见结论） |
| TF-IDF 拟合范围 | 只在训练集 fit | 同（无差异） | 防泄漏的做法原版是对的，直接复用该原则 |
| 标签编解码 | `LabelEncoder` 编码为 0..n-1，持久化 encoder | 不引入 LabelEncoder，字符串标签直接交给 sklearn | 少一层编解码就少一个配对出错点（原版 predictor 曾把词表词当标签，正是配对层引入的 bug）。代价：标签列是字符串、存储略大；下游要数值 id 时需自行映射 |
| 模型数量与选型 | 训 4 个模型（NB/LR/SVM/Forest），验证集 F1 选优 | 只训 logistic，无选型逻辑 | 重写的目的是吃透流程而非堆模型；W02 受控实验再按需扩展 |
| 产物存放位置 | `models/`（与环境共享，磁盘上混有 sklearn 1.6.1 存的旧文件） | `rebuilt/artifacts/` 独立目录 | 旧 pkl 由 1.5.2 环境加载产生 InconsistentVersionWarning（实证）；混放会互相污染，独立目录隔离实验产物 |
| 预测接口 | `_find_best_model` 按硬编码顺序取第一个文件，注释却写"F1 最高" | 加载固定产物，无选型；选型结果由训练流程保证 | 硬编码顺序是把选型决定藏在了加载侧；rebuilt 的分工是"训练侧定，预测侧只执行" |
| 错误处理 | 混合（有 raise 有 print 继续） | 显式抛 `FileNotFoundError`；目录用 `exist_ok=True` | 静默继续会把错误推迟到更难查的下游；边界处早失败 |

## 值得对比的点（自己判断，不要都写成"无差异"）

- **sys.path.insert vs 目录内运行**：sys.path.insert 解决的是"从别处运行也能 import"，但代价是每个入口都要记得注入、且全局生效。rebuilt 5 个文件都在同一目录，运行目录即包路径，不需要这层。等真长成多包项目，正确解法是打包（`pip install -e .`），而不是 path 注入。
- **LabelEncoder vs 字符串标签**：sklearn 分类器原生支持字符串 y，encoder 只在"需要数值 id"（如 embedding、神经网络）时才有必要。原版需要 encoder 还因为要持久化它供预测端逆变换——这本身又多了一个"必须同时成功"的产物。
- **产物分开存放**：`models/` 是全项目共享的，旧文件是另一个 sklearn 版本存的；如果 rebuilt 也写那里，`InconsistentVersionWarning` 会再次出现且无法区分是谁的产物。独立 `artifacts/` 让"这批产物的环境"有明确边界。
- **vectorizer 与 model 分两个文件的隐患**：两个文件可能不同批（一个重训了一个没重训），加载端无从察觉——这正是 W02 要做 manifest 一致性校验（记录 sklearn 版本、数据指纹、样本数，不匹配即报错）的动机。
- **rebuilt 避开了 `_find_best_model` 的坑吗**：避开了——rebuilt 没有"挑文件"逻辑，加载路径固定。但"哪个是最佳"目前只靠流程约定（训练侧选好才存），正式的机制化要等 W02 的 manifest。
- **emoji print vs 纯文本**：有关系。Windows 默认 GBK 控制台打印 emoji 会抛 `UnicodeEncodeError`，脚本中断，复现性被一个 print 破坏。纯文本（或显式设置 stdout 编码）是复现性细节的一部分。

## 结论

- 重写版验证集 macro-F1：**0.9950**（对比基线：logistic 0.9975）
- 与原版是否有实质行为差异：**无**。差的 1 条已实证：全量 4 类数据含 5 对"同文本不同标签"的 cross-post，其中 1 对跨 train/val，即 400 条验证集中 1 条分歧 = 0.0025。数据指纹 sha256 前 16 位 `374fa3d7bfd13759`。
- 下一步（W02 要做的事）：
  1. 重复样本受控实验：保留 vs 去重后按组切分（cross-post 不得跨集）
  2. 选型口径统一：trainer / evaluate.py / predictor 三处统一为"验证集选型"，predictor 去掉硬编码顺序
  3. manifest 一致性校验：sklearn 版本 + 数据指纹 + 样本数，不匹配报错；用当前环境重训覆盖旧 1.6.1 产物
