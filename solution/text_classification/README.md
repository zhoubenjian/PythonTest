# 文本分类项目 (Text Classification)

基于 sklearn 的 20_newsgroups 数据集，实现完整的文本分类 MLOps 流水线，支持多模型训练、评估、预测和交付。

## 📁 项目结构

```
text_classification/
├── configs/
│   └── config.py               # 全局配置（路径、超参数、模型列表）
├── src/
│   ├── data_loader.py         # 数据加载（本地目录 → DataFrame）
│   ├── preprocessor.py        # 文本预处理（TF-IDF + LabelEncoder）
│   ├── model.py               # 模型工厂（动态创建 sklearn 分类器）
│   ├── trainer.py             # 训练器（训练 + 保存）
│   ├── evaluator.py           # 评估器（指标 + 对比 + 报告）
│   └── predictor.py           # 预测器（单条/批量/交互式）
├── scripts/
│   ├── train.py               # 训练入口
│   ├── evaluate.py            # 评估入口
│   └── predict.py             # 预测入口
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   └── test_predictor.py
├── data/
│   ├── raw/                   # 原始数据（.csv）
│   └── processed/             # 处理后数据（.npz）
├── models/                    # 交付物（.pkl）
├── README.md
└── STEP.md                    # 详细工作原理说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn joblib scipy
```

### 2. 准备数据

将 20_newsgroups 数据集放置到 `local_datas/20_newsgroups/` 目录下。

目录结构示例：
```
local_datas/20_newsgroups/
├── alt.atheism/
├── comp.graphics/
├── comp.os.ms-windows.misc/
├── rec.sport.baseball/
└── ...
```

### 3. 训练模型

```bash
cd text_classification
python scripts/train.py --save
```

训练完成后 `models/` 目录会生成：
```
models/
├── preprocessor.pkl          # 预处理器（TF-IDF + LabelEncoder）
├── naive_bayes_model.pkl     # 朴素贝叶斯
├── logistic_model.pkl        # 逻辑回归
├── svm_model.pkl             # SVM
└── forest_model.pkl          # 随机森林
```

### 4. 评估模型

```bash
python scripts/evaluate.py              # 评估所有模型
python scripts/evaluate.py --model svm   # 只评估 SVM
```

### 5. 预测

```bash
python scripts/predict.py                           # 交互式模式
python scripts/predict.py --text "God is love"      # 单条预测
python scripts/predict.py --batch                   # 批量 demo
python scripts/predict.py --model svm --text "..."  # 指定模型
```

### 6. 运行测试

```bash
python -m unittest discover tests/ -v
```

## 📦 交付给甲方

交付清单：
```
models/
├── preprocessor.pkl    # 预处理器（必须）
└── xxx_model.pkl       # 选定的模型（如 svm_model.pkl）
```

甲方使用：
```python
from src.predictor import TextClassifierPredictor

predictor = TextClassifierPredictor(
    model_path='models/svm_model.pkl',
    preprocessor_path='models/preprocessor.pkl',
)
result = predictor.predict("要分类的文本")
print(result)
# {'category': 'comp.graphics', 'confidence': 0.92, 'probabilities': {...}}
```

## 🎯 支持的模型

| key | 类名 | 说明 |
|-----|------|------|
| `naive_bayes` | `MultinomialNB` | 朴素贝叶斯 |
| `logistic` | `LogisticRegression` | 逻辑回归 |
| `svm` | `SVC` | 支持向量机（linear kernel） |
| `forest` | `RandomForestClassifier` | 随机森林 |

在 `configs/config.py` 的 `MODEL_CONFIGS` 中修改超参数或新增模型。

## 📊 数据说明

- 数据集：20_newsgroups（新闻组文本分类）
- 默认类别：4 类（alt.atheism / comp.graphics / comp.os.ms-windows / rec.sport.baseball）
- 数据划分：训练 80% / 验证 10% / 测试 10%
- 特征提取：TF-IDF（max_features=10000, ngram_range=(1,2)）