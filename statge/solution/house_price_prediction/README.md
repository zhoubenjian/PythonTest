# 🏠 房价预测项目 (House Price Prediction)

基于 California Housing 数据集的房价预测系统，采用标准工程化结构设计。

## 📁 项目结构

```
house_price_prediction/
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后的数据
├── src/                     # 源代码
│   ├── data_loader.py       # 数据加载
│   ├── preprocessor.py      # 数据预处理
│   ├── model.py             # 模型定义
│   ├── trainer.py           # 训练器
│   ├── evaluator.py         # 评估器
│   └── predictor.py         # 预测器
├── configs/                 # 配置
│   └── config.py            # 全局配置
├── scripts/                 # 脚本
│   ├── train.py             # 训练
│   ├── evaluate.py          # 评估
│   └── predict.py           # 预测
└── tests/                   # 测试
    ├── test_data_loader.py
    ├── test_preprocessor.py
    └── test_model.py
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install numpy pandas scikit-learn joblib
```

### 2. 训练模型

```bash
# 训练所有模型
python scripts/train.py --save

# 训练指定模型
python scripts/train.py --model ridge --save

# 训练神经网络模型
python scripts/train.py --model nn --save
```

### 3. 评估模型

```bash
# 评估所有已保存的模型
python scripts/evaluate.py

# 评估指定模型
python scripts/evaluate.py --model-path models/ridge_model.pkl
```

### 4. 预测房价

```bash
# 交互式预测
python scripts/predict.py

# 批量演示
python scripts/predict.py --batch

# 指定模型
python scripts/predict.py --model models/forest_model.pkl
```

### 5. 运行测试

```bash
# 运行所有测试
python tests/test_data_loader.py
python tests/test_preprocessor.py
python tests/test_model.py
```

## 🧠 支持的模型

| 模型 | 类型 | 说明 |
|------|------|------|
| `linear` | LinearRegression | 线性回归 |
| `ridge` | Ridge | 岭回归（L2 正则化） |
| `forest` | RandomForestRegressor | 随机森林 |
| `nn` | MLPRegressor | 多层感知机神经网络 |

## 📊 数据集

**California Housing Dataset**
- 来源: sklearn.datasets.fetch_california_housing
- 样本数: 20,640 条
- 特征数: 8 维
- 目标: 房价中位数（单位：万美元）

### 特征说明

| 特征 | 说明 |
|------|------|
| MedInc | 收入中位数 |
| HouseAge | 房龄 |
| AveRooms | 平均房间数 |
| AveBedrms | 平均卧室数 |
| Population | 人口 |
| AveOccup | 平均入住率 |
| Latitude | 纬度 |
| Longitude | 经度 |

## 📈 评估指标

- **MSE**: 均方误差 (Mean Squared Error)
- **RMSE**: 均方根误差 (Root Mean Squared Error)
- **MAE**: 平均绝对误差 (Mean Absolute Error)
- **R²**: 决定系数 (Coefficient of Determination，越接近 1 越好)

## 🔧 配置说明

在 `configs/config.py` 中可以调整：

- 数据划分比例（`TRAIN_SIZE`, `VAL_SIZE`）
- 随机种子（`RANDOM_STATE`）
- 模型超参数（`MODEL_CONFIGS`）

## 📝 架构设计

```
┌─────────────┐
│  scripts/   │  入口脚本（训练/评估/预测）
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   src/      │  核心模块
│             │
│ ┌─────────┐ │
│ │data_loader│ │  数据加载
│ └─────────┘ │
│ ┌─────────┐ │
│ │preprocessor│ │  数据预处理
│ └─────────┘ │
│ ┌─────────┐ │
│ │  model   │ │  模型定义
│ └─────────┘ │
│ ┌─────────┐ │
│ │ trainer  │ │  训练器
│ └─────────┘ │
│ ┌─────────┐ │
│ │evaluator │ │  评估器
│ └─────────┘ │
│ ┌─────────┐ │
│ │predictor │ │  预测器
│ └─────────┘ │
└─────────────┘
       │
       ▼
┌─────────────┐
│ configs/    │  配置管理
└─────────────┘
```

## 📄 License

MIT