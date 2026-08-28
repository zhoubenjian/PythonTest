'''
数据预处理模块测试
'''
import os
import sys
import numpy as np

# 添加项目根目录和 src 目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.data_loader import get_demo_data
from src.preprocessor import DataPreprocessor


def test_prepare_data():
    """测试数据预处理"""
    df = get_demo_data(n_samples=200)
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    assert "X_train" in data_dict
    assert "y_test" in data_dict
    assert data_dict["X_train"].shape[0] > 0
    assert data_dict["X_test"].shape[0] > 0
    print(f"✅ test_prepare_data 通过")
    print(f"   训练集: {data_dict['X_train'].shape}")
    print(f"   测试集: {data_dict['X_test'].shape}")

def test_inverse_transform():
    """测试反标准化"""
    df = get_demo_data(n_samples=200)
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # 反标准化测试
    y_scaled = data_dict["y_train"][:5]
    y_original = preprocessor.inverse_transform_y(y_scaled)

    assert len(y_original) == 5
    assert np.all(np.isfinite(y_original))
    print(f"✅ test_inverse_transform 通过: {y_original[:3]}")


if __name__ == "__main__":
    print("🧪 数据预处理模块测试")
    print("=" * 50)
    test_prepare_data()
    test_inverse_transform()
    print("\n🎉 所有预处理测试通过！")