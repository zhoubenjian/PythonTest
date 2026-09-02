# House Price Prediction 项目工作原理

## 一、整体架构

```
用户调用入口脚本 (scripts/)
        │
        ▼
   核心模块 (src/)
        │
        ▼
   配置管理 (configs/)
```

项目遵循典型的 MLOps 流水线设计，分为 5 个阶段：

```
数据加载 => 数据预处理 => 模型训练 => 模型评估 => 模型预测
```

---

## 二、各文件作用详解

### `configs/config.py` — 全局配置中心

- **路径配置**：定义 `PROJECT_ROOT`、`DATA_DIR`、`RAW_DATA_DIR`、`PROCESSED_DATA_DIR` 等路径，导入时自动 `mkdir` 创建目录
- **数据配置**：指定使用 `california_housing` 数据集，定义 8 个特征列和 1 个目标列
- **数据划分**：`TRAIN_SIZE=0.8`、`VAL_SIZE=0.1`、测试集 0.1，随机种子 42
- **模型配置**：集中定义 4 种模型的超参数（线性回归、岭回归、随机森林、MLP 神经网络）

### `src/data_loader.py` — 数据加载器

| 函数 | 作用 |
|------|------|
| `load_raw_data()` | 调用 sklearn 的 `fetch_california_housing()` 在线下载 California 住房数据 |
| `save_raw_data(df)` | 将原始数据保存为 CSV 到 `data/raw/` |
| `load_from_csv(filepath)` | 从本地 CSV 文件加载数据 |
| `get_demo_data(n)` | 取前 n 行作为演示数据 |

> ⚠️ 注意：当前流水线只调用了 `load_raw_data()`，没有调用 `save_raw_data()`，所以 `data/raw/` 是空的。

### `src/preprocessor.py` — 数据预处理器

`DataPreprocessor` 类完成以下工作：

1. **特征/标签分离**：`X = df[FEATURE_COLUMNS]`，`y = df[TARGET_COLUMN]`
2. **数据划分**：先用 `train_test_split` 划出 90% (训练+验证) 和 10% (测试)，再从 90% 中划出 10% 作为验证集
3. **标准化**：对 X 和 y 分别做 `StandardScaler`，使得特征均值为 0、方差为 1
4. **保存**：将处理后的数据以 `.npz` 格式压缩保存到 `data/processed/`

> 关键设计：同时保留了原始尺度的 X_train/val/test（`X_train_raw` 等）供反标准化使用。

### `src/model.py` — 模型工厂

`get_model(model_type)` 根据配置字符串动态创建 sklearn 模型：

```python
# config 中的 name 字段 => sklearn 类
"LinearRegression" => LinearRegression()
"Ridge"            => Ridge(alpha=0.1)
"RandomForestRegressor" => RandomForestRegressor(n_estimators=100, max_depth=10)
"MLPRegressor"      => MLPRegressor(hidden_layer_sizes=(128,64,32), ...)
```

通过 `globals().get(model_name)` 反射查找类，实现配置驱动的模型创建，新增模型只需改 config 即可。

### `src/trainer.py` — 模型训练器

`ModelTrainer` 类：

| 方法 | 作用 |
|------|------|
| `train(model_type, data_dict)` | 训练单个模型：创建模型 => `fit()` => 计算训练/验证集指标 => 存入 `self.models` |
| `train_all(data_dict)` | 遍历 `MODEL_CONFIGS` 训练所有模型，并打印性能对比表 |
| `save_model()` / `save_all_models()` | 用 `joblib.dump()` 保存模型到 `.pkl` 文件 |
| `_calculate_metrics()` | 计算 MSE、RMSE、MAE、R² 四个指标 |

### `src/evaluator.py` — 模型评估器

`ModelEvaluator` 类：

| 方法 | 作用 |
|------|------|
| `evaluate(model, X, y)` | 单模型评估，返回 MSE/RMSE/MAE/R² 和预测值 |
| `compare_models(models_dict, X_test, y_test)` | 多模型对比，打印性能表格并找出 R² 最高的最佳模型 |
| `get_error_analysis(y_true, y_pred)` | 误差分析：均值误差、最大/最小误差、高估/低估次数等 |

### `src/predictor.py` — 预测器

`HousePricePredictor` 类：

| 方法 | 作用 |
|------|------|
| `predict(features)` | 支持 dict/DataFrame/ndarray 三种输入 => 标准化 => 预测 => **反标准化回原始尺度** |
| `predict_single(**kwargs)` | 单样本预测，关键字参数如 `MedInc=5.0` |
| `predict_batch(feature_list)` | 批量预测，传入字典列表 |

> 关键：预测时必须用训练时的 `scaler_X` 和 `scaler_y` 做相同的标准化/反标准化处理。

### `scripts/train.py` — 训练入口

```
Step 1: load_raw_data()          加载数据
Step 2: DataPreprocessor         预处理 + 保存 .npz
Step 3: ModelTrainer.train()     训练模型
Step 4: joblib.dump()            保存模型（--save 参数）
```

### `scripts/evaluate.py` — 评估入口

```
Step 1: 加载已处理数据 / 重新预处理
Step 2: joblib.load() 加载模型
Step 3: ModelEvaluator.compare_models()  对比评估
```

### `scripts/predict.py` — 预测入口

```
Step 1: joblib.load() 加载模型（自动选最佳模型或指定模型）
Step 2: DataPreprocessor 重新拟合预处理器
Step 3: interactive_mode() 或 batch_demo()  交互式/批量预测
```

---

## 三、数据流示意图

```
┌─────────────────────┐
│ fetch_california_   │  ← sklearn 在线下载
│ housing()           │
└─────────┬───────────┘
          │ pd.DataFrame
          ▼
┌─────────────────────┐
│  DataPreprocessor   │
│  ┌───────────────┐  │
│  │ train_test_   │  │  80%/10%/10%
│  │ split × 2     │  │
│  └───────┬───────┘  │
│          ▼          │
│  ┌───────────────┐  │
│  │ StandardScaler│  │  X 和 y 分别标准化
│  └───────┬───────┘  │
└─────────┬───────────┘
          │ data_dict (6 个数组)
          ▼
┌─────────────────────┐
│   ModelTrainer      │  .fit() => 保存 .pkl
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌──────────┐ ┌──────────┐
│ Evaluator│ │Predictor │  反标准化 => 原始尺度
└──────────┘ └──────────┘
```

---

## 四、设计亮点

1. **配置驱动**：所有超参数集中在 `config.py`，修改无需改动业务代码
2. **工厂模式**：`model.py` 用反射动态创建模型，扩展新模型只需加配置
3. **数据标准化**：特征和标签都做标准化（`StandardScaler`），提升模型收敛速度
4. **反标准化预测**：预测时通过 `inverse_transform_y()` 将标准化的预测值还原为真实房价
5. **三划分策略**：训练/验证/测试 8:1:1，验证集用于调参，测试集用于最终评估

---

## 五、项目结构

```
house_price_prediction/
├── configs/
│   └── config.py            # 全局配置
├── data/
│   ├── raw/                 # 原始数据（当前为空，需手动保存）
│   └── processed/           # 处理后的数据 (.npz)
├── models/
│   └── ridge_model.pkl      # 已保存的模型
├── scripts/
│   ├── train.py             # 训练入口
│   ├── evaluate.py          # 评估入口
│   └── predict.py           # 预测入口
├── src/
│   ├── data_loader.py       # 数据加载
│   ├── preprocessor.py      # 数据预处理
│   ├── model.py             # 模型工厂
│   ├── trainer.py           # 训练器
│   ├── evaluator.py         # 评估器
│   └── predictor.py         # 预测器
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_model.py
└── README.md
```