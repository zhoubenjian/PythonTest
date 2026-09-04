顺序 1: configs/config.py          ← 无 __main__，自动导入验证
顺序 2: src/model.py               ← 独立，创建 4 个模型
顺序 3: src/data_loader.py         ← 加载本地数据
顺序 4: src/preprocessor.py        ← 依赖 data_loader
顺序 5: src/trainer.py             ← 依赖 preprocessor + model
顺序 6: src/evaluator.py           ← 依赖 trainer
顺序 7: src/predictor.py           ← 依赖 trainer.py 跑完的模型文件



# 文本分类项目 — 工作原理详解

## 一、整体架构

```
scripts/ (入口)
  │
  ▼
src/ (核心模块)
  │
  ▼
configs/config.py (配置中心)
```

MLOps 流水线：**数据加载 → 文本预处理 → 模型训练 → 模型评估 → 模型预测**

---

## 二、各文件作用

### `configs/config.py` — 全局配置中心

| 配置项 | 说明 |
|--------|------|
| `PROJECT_ROOT` | 项目根目录，自动计算 |
| `DATA_DIR` / `RAW_DATA_DIR` / `PROCESSED_DATA_DIR` | 数据路径 |
| `MODEL_DIR` / `PREPROCESSOR_PATH` | 模型交付路径 |
| `LOCAL_DATA_PATH` | 本地 20_newsgroups 数据路径 |
| `DEFAULT_CATEGORIES` | 默认 4 个类别 |
| `TRAIN_SIZE` / `VAL_SIZE` | 数据划分比例（0.8 / 0.1） |
| `TFIDF_CONFIG` | TF-IDF 超参数 |
| `MODEL_CONFIGS` | 4 个分类器的超参数 |
| `AVERAGE_METHOD` | 评估时 macro 平均 |

> 导入 config.py 时会自动 `os.makedirs()` 创建所有目录。

### `src/data_loader.py` — 数据加载器

从本地目录加载 20_newsgroups 数据（使用 `sklearn.datasets.load_files`）：

```
load_files(container_path=LOCAL_DATA_PATH)
  → data.data (原始文本)
  → data.target (类别 id)
  → data.target_names (类别名称)
```

| 函数 | 作用 |
|------|------|
| `load_raw_data(categories)` | 加载本地数据，过滤指定类别 |
| `save_raw_data(df)` | 保存为 CSV 到 `data/raw/` |
| `load_from_csv(filepath)` | 从 CSV 加载 |

### `src/preprocessor.py` — 文本预处理器

`TextPreprocessor` 类完成以下工作：

```
原始文本 → train_test_split(8:1:1) → TfidfVectorizer.fit_transform()
                                    → LabelEncoder.fit_transform()
```

**为什么同时需要 save 两种文件？**

| 文件 | 内容 | 用途 |
|------|------|------|
| `models/preprocessor.pkl` | TfidfVectorizer + LabelEncoder 对象 | 预测新文本时：把新文本转 TF-IDF，把数字转回类别名 |
| `data/processed/*.npz` | 已转换好的 X/y 稀疏矩阵和数组 | 评估时：直接加载跳过重新划分和转换 |

**稀疏矩阵的保存方式**（踩过的坑）：

```python
# 错误：np.savez_compressed 会把 csr_matrix 包装成 0-d object array
np.savez_compressed('X.npz', X=sparse_matrix)

# 正确：用 scipy.sparse 专用 API
scipy.sparse.save_npz('X.npz', sparse_matrix)
scipy.sparse.load_npz('X.npz')
```

### `src/model.py` — 模型工厂

`get_model(model_type)` 根据配置字符串动态创建 sklearn 模型：

```
"naive_bayes" → MultinomialNB(alpha=0.1)
"logistic"    → LogisticRegression(C=1.0, max_iter=1000)
"svm"         → SVC(C=1.0, kernel='linear', probability=True)  ← 必须 probability=True 才有 predict_proba
"forest"      → RandomForestClassifier(n_estimators=100, max_depth=20)
```

通过遍历 sklearn 的子模块（naive_bayes、linear_model、svm、ensemble）用反射查找类。

### `src/trainer.py` — 模型训练器

`ModelTrainer` 类：

| 方法 | 作用 |
|------|------|
| `train(model_type, data_dict)` | 训练单个模型，存入 `self.models` |
| `train_all(data_dict)` | 遍历 MODEL_CONFIGS 训练所有 |
| `save_all_models()` | `joblib.dump()` 保存所有 |
| `save_best_model()` | 只保存验证集 F1 最高的 |

**训练指标**：accuracy + F1(macro)

### `src/evaluator.py` — 模型评估器

`ModelEvaluator` 类：

| 方法 | 作用 |
|------|------|
| `evaluate(model, X, y)` | 单模型评估，返回 accuracy/f1/precision/recall |
| `print_report(model, X, y)` | 打印详细分类报告 |
| `print_confusion_matrix(model, X, y)` | 打印混淆矩阵 |
| `compare_models(models_dict, X_test, y_test)` | 多模型对比，返回 F1 最高的 |
| `get_error_analysis(y_true, y_pred)` | 找出错误分类的样本 |

### `src/predictor.py` — 预测器

`TextClassifierPredictor` 类：

```python
predictor = TextClassifierPredictor(model_path, preprocessor_path)
result = predictor.predict("God is love and religion...")
# {'category': 'alt.atheism', 'confidence': 0.95, 'probabilities': {...}}
```

| 方法 | 作用 |
|------|------|
| `predict(text)` | 单条预测，返回类别 + 置信度 + 各类别概率 |
| `predict_batch(texts)` | 批量预测 |
| `interactive()` | 交互式命令行模式 |

**预测时的数据流向**：

```
新文本 (str)
  → preprocessor.transform_text()    # TF-IDF 稀疏矩阵
  → model.predict_proba()            # 各类别概率
  → preprocessor.decode_label()      # 数字 → 类别名
  → {category, confidence, probabilities}
```

### `scripts/train.py` — 训练入口

```
Step 1: load_raw_data()          加载本地 20_newsgroups 数据
Step 2: TextPreprocessor         划分 + TF-IDF + LabelEncoder
Step 3: ModelTrainer.train_all() 训练 4 个模型
Step 4: preprocessor.save() + save_processed_data() + trainer.save_all_models()
         （--save 参数触发）
```

### `scripts/evaluate.py` — 评估入口

```
Step 1: TextPreprocessor.load()        加载预处理器
Step 2: 扫描 models/ 下所有 .pkl 模型文件
Step 3: load_processed_data()           加载 .npz 数据（跳过重新划分！）
Step 4: ModelEvaluator.compare_models() 测试集对比 + 分类报告 + 混淆矩阵
```

### `scripts/predict.py` — 预测入口

```bash
python scripts/predict.py                      # 交互式
python scripts/predict.py --text "..."         # 单条
python scripts/predict.py --batch              # 批量 demo
python scripts/predict.py --model svm --text   # 指定模型
```

---

## 三、数据流示意图

```
┌─────────────────────────────────────┐
│ load_files(LOCAL_DATA_PATH)        │ ← 本地目录
│ → DataFrame(text, category)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ TextPreprocessor.prepare_data()     │
│ ┌───────────────────────────────┐  │
│ │ train_test_split × 2          │  │  80% / 10% / 10%
│ │   (stratify=类别)             │  │
│ └───────────────┬───────────────┘  │
│                 ▼                  │
│ ┌───────────────────────────────┐  │
│ │ TfidfVectorizer.fit_transform │  │  max_features=10000
│ │ ngram_range=(1,2)             │  │  stop_words='english'
│ └───────────────┬───────────────┘  │
│                 ▼                  │
│ ┌───────────────────────────────┐  │
│ │ LabelEncoder.fit_transform    │  │  alt.atheism→0
│ │                               │  │  comp.graphics→1
│ └───────────────┬───────────────┘  │
└─────────────────┼───────────────────┘
                  │ data_dict (稀疏矩阵 + 数组)
                  ▼
┌─────────────────────────────────────┐
│ ModelTrainer.train_all()           │
│ MultinomialNB / LogisticRegression │
│ SVC / RandomForestClassifier       │
│   → fit() → 存入 self.models       │
└─────────────────┬───────────────────┘
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
┌───────────┐             ┌───────────────┐
│ Evaluator │             │   Predictor   │
│ 对比评估  │             │ 预测新文本     │
│ 分类报告  │             │ 交互式/批量    │
│ 混淆矩阵  │             │               │
└───────────┘             └───────────────┘
```

---

## 四、设计亮点

1. **配置驱动**：所有超参数集中在 `configs/config.py`，新增模型只需加一行配置
2. **工厂模式**：`model.py` 用反射动态创建模型，`get_model('svm')` 一行搞定
3. **稀疏矩阵正确持久化**：使用 `scipy.sparse.save_npz` 而非 `np.savez`，避免 0-d array bug
4. **预处理器独立保存**：`preprocessor.pkl` 让甲方拿到就能预测，不需要原始数据
5. **processed_data 加速评估**：`.npz` 让 `evaluate.py` 跳过重新划分，直接加载
6. **三划分策略**：训练/验证/测试 8:1:1，验证集用于选最佳超参数，测试集用于最终评估
7. **统一 save/load API**：model 和 preprocessor 都有 `save()` / `load()` 方法
8. **`__main__` 自测**：每个 src 模块都能独立运行验证

---

## 五、与房价预测项目的对比

| 对比项 | house_price_prediction | text_classification |
|--------|----------------------|---------------------|
| 数据源 | sklearn 在线下载 | 本地目录 load_files |
| 特征处理 | StandardScaler | TfidfVectorizer |
| 标签处理 | StandardScaler (y) | LabelEncoder |
| 输出类型 | 密集 numpy 数组 | 稀疏矩阵 (csr_matrix) |
| 反标准化 | inverse_transform_y | decode_label |
| 模型数 | 4 个回归器 | 4 个分类器 |
| 评估指标 | MSE / RMSE / MAE / R² | Accuracy / F1 / Precision / Recall |
| 数据保存 | np.savez_compressed | scipy.sparse.save_npz + np.savez |

---

## 六、交付给甲方

### 交付清单

```
text_classification/
├── models/
│   ├── preprocessor.pkl       ← 必须（TF-IDF + LabelEncoder）
│   └── svm_model.pkl          ← 选定的模型
├── src/
│   └── predictor.py           ← 预测入口（或封装成 API）
├── configs/
│   └── config.py              ← 配置（可选）
└── README.md                  ← 使用说明
```

### 甲方使用

```bash
pip install pandas numpy scikit-learn joblib scipy
```

```python
from src.predictor import TextClassifierPredictor

predictor = TextClassifierPredictor()
result = predictor.predict("This text is about computer graphics")
print(result['category'], result['confidence'])
```

### 可选：封装为 API

```python
from flask import Flask, request, jsonify
from src.predictor import TextClassifierPredictor

app = Flask(__name__)
predictor = TextClassifierPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    result = predictor.predict(text)
    return jsonify(result)

# POST http://localhost:5000/predict
# body: {"text": "God is love"}
```