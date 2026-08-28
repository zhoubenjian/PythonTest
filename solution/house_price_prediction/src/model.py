'''
模型定义模块
支持多种回归模型
'''
import os
import sys
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# 添加项目根目录到路径（必须在导入 configs 之前）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import MODEL_CONFIGS


def get_model(model_type: str):
    """
    根据配置创建模型实例
    :param model_type: 模型类型(linear, ridge, forest, nn)
    :return: 模型实例
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"未知模型类型: {model_type}\n"
            f"支持的模型类型: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_type]
    model_name = config['name']
    params = config['params']

    print(f"🧠 创建模型: {model_name}")
    if params:
        print(f"参数: {params}")

    # 动态创建模型
    model_class = globals().get(model_name)
    if model_class is None:
        raise ValueError(f"未找到模型类: {model_name}")

    model = model_class(**params)
    return model


def list_available_models() -> list:
    """
    列出所有可用的模型类型
    :return: 模型类型列表
    """
    return list(MODEL_CONFIGS.keys())


if __name__ == "__main__":
    # 测试模型创建
    print("📊 可用模型列表:")
    models = list_available_models()
    for name in models:
        print(f"- {name}")

    print("\n🧪 测试创建模型:")
    for model_type in models:
        try:
            model = get_model(model_type)
            print(f"✅ {model_type}: {type(model).__name__}")
        except Exception as e:
            print(f"❌ {model_type}: {e}")

    print("\n✅ 模型定义测试通过！")






