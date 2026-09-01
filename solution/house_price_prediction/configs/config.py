'''
全局配置文件
定义项目中使用的超参数和路径
'''
import os

import sys
# 添加项目根目录到路径，解决模块导入问题
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


'''
========== 路径配置 ==========
'''
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 模型路径
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# 确保目录存在
os.makedirs(PROJECT_ROOT, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


'''
========== 数据配置 ==========
'''
# 使用California Housing 数据集(sklearn内置)
DATASET_NAME = 'california_housing'

# 特征列表
FEATURE_COLUMNS = [
    "MedInc",       # 收入中位数
    "HouseAge",     # 房龄
    "AveRooms",     # 平均房间数
    "AveBedrms",    # 平均卧室数
    "Population",   # 人口
    "AveOccup",     # 平均入住率
    "Latitude",     # 纬度
    "Longitude"     # 经度
]
TARGET_COLUMN = "MedHouseVal"  # 目标：房价中位数


'''
========== 数据划分 ==========
'''
# 随机种子
RANDOM_STATE = 42
# 训练集比例
TRAIN_SIZE = 0.8
# 验证集比例10%：调整超参数、选择模型、防止过拟合，训练过程中多次使用。(剩余为测试集10%：最终评估模型泛化能力，训练完成后一次性使用。)
VAL_SIZE = 0.1


'''
========== 模型配置 ==========
'''
MODEL_CONFIGS = {
    # 线性回归（基线模型）
    # 特点：无正则化，最简单，用作性能基准
    # 适用场景：数据线性可分，特征之间无多重共线性
    "linear": {
        "name": "LinearRegression",
        "params": {}        # 无超参数，使用默认设置
    },

    # 岭回归（L2 正则化线性模型）
    # 特点：通过 L2 惩罚控制过拟合，系数趋近于 0 但不为 0
    # 适用场景：特征较多、存在多重共线性，或特征数 > 样本数
    "ridge": {
        "name": "Ridge",
        "params": {
            "alpha": 0.1,   # 正则化强度（越大惩罚越强，模型越简单）
        }
    },

    # 随机森林回归（集成树模型）
    # 特点：基于 Bagging 思想，多棵决策树投票，抗过拟合能力强
    # 适用场景：非线性关系、特征交互复杂、需要特征重要性分析
    "forest": {
        "name": "RandomForestRegressor",
        "params": {
            "n_estimators": 100,    # 决策树的数量（越多越稳定，但计算成本增加）
            "max_depth": 10,        # 每棵树的最大深度（控制过拟合，值越大越复杂）
            "random_state": RANDOM_STATE
        }
    },


    # 神经网络回归（MLP 多层感知机）
    # 特点：深度学习的入门模型，能拟合任意复杂函数
    # 适用场景：大数据量、高度非线性、特征工程复杂的问题
    "nn": {
        "name": "MLPRegressor",
        "params": {
            "hidden_layer_sizes": (128, 64, 32),    # 隐藏层结构：3层(第一层:128 => 第二层:64 => 第三层:32)，神经元数逐层递减
            "activation": "relu",                   # ReLU激活函数 能缓解梯度消失，加速收敛
            "solver": "adam",                       # Adam优化器 自适应学习率，适合大多数情况
            "early_stopping": True,                 # 启用早停法(验证集性能不再提升时提前终止训练)
            "random_state": RANDOM_STATE
        }
    }
}


'''
========== 评估指标 ==========
'''
METRICS = ['mse', 'rmse', 'mae', 'r2']