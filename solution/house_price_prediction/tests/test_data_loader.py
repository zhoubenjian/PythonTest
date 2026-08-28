'''
数据加载模块测试
'''
import os
import sys

# 添加项目根目录和 src 目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.data_loader import load_raw_data, get_demo_data

def test_load_raw_data():
    """测试加载原始数据"""
    df = load_raw_data()
    assert df is not None, "DataFrame 不应为空"
    assert df.shape[0] > 0, "应该有数据行"
    assert df.shape[1] == 9, "应该有 9 列 (8 特征 + 1 目标)"
    print(f"✅ test_load_raw_data 通过: {df.shape}")

def test_get_demo_data():
    """测试获取演示数据"""
    n_samples = 50
    df = get_demo_data(n_samples=n_samples)
    assert len(df) == n_samples, f"应该返回 {n_samples} 条数据"
    print(f"✅ test_get_demo_data 通过: {df.shape}")


if __name__ == "__main__":
    print("🧪 数据加载模块测试")
    print("=" * 50)
    test_load_raw_data()
    test_get_demo_data()
    print("\n🎉 所有数据加载测试通过！")