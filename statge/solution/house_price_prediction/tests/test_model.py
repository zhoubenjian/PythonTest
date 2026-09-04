'''
模型模块测试
'''
import os
import sys

# 添加项目根目录和 src 目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.model import get_model, list_available_models
from configs.config import MODEL_CONFIGS


def test_list_models():
    """测试列出可用模型"""
    models = list_available_models()
    assert len(models) > 0, "应该有可用模型"
    print(f"✅ test_list_models 通过: {models}")

def test_create_models():
    """测试创建各种模型"""
    for model_type in MODEL_CONFIGS:
        model = get_model(model_type)
        assert model is not None, f"{model_type} 模型创建失败"
        print(f"✅ {model_type} 模型创建成功: {type(model).__name__}")

def test_model_training():
    """测试模型训练流程"""
    from src.data_loader import get_demo_data
    from src.preprocessor import DataPreprocessor
    from src.trainer import ModelTrainer

    # 准备数据
    df = get_demo_data(n_samples=200)
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # 训练单个模型
    trainer = ModelTrainer()
    result = trainer.train("ridge", data_dict)

    assert "val_metrics" in result
    assert result["val_metrics"]["r2"] > -10, "R² 应该合理"
    print(f"✅ test_model_training 通过")
    print(f"   验证集 R²: {result['val_metrics']['r2']:.4f}")


if __name__ == "__main__":
    print("🧪 模型模块测试")
    print("=" * 50)
    test_list_models()
    test_create_models()
    test_model_training()
    print("\n🎉 所有模型测试通过！")