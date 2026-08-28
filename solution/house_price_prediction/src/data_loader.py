'''
数据加载模块
从数据源加载数据
'''
import os
import sys
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 添加项目根目录到路径（必须在导入 configs 之前）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import RAW_DATA_DIR, DATASET_NAME


def load_raw_data() -> pd.DataFrame:
    """
    加载 California Housing 原始数据集
    Returns:
        pd.DataFrame: 包含特征和标签的 DataFrame
    """
    print("📥 正在加载 California Housing 数据集...")

    # 加载数据
    housing = fetch_california_housing(as_frame=True)

    # 构造 DataFrame
    df = housing.frame

    print(f"✅ 数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"   特征: {list(df.columns[:-1])}")
    print(f"   目标: {df.columns[-1]}")

    return df

def save_raw_data(df: pd.DataFrame) -> str:
    """
    将原始数据保存到本地
    Args:
        df: 原始数据 DataFrame
    Returns:
        str: 保存的文件路径
    """
    filepath = os.path.join(RAW_DATA_DIR, f"{DATASET_NAME}_raw.csv")
    df.to_csv(filepath, index=False)
    print(f"💾 原始数据已保存: {filepath}")
    return filepath

def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    从 CSV 文件加载数据
    Args:
        filepath: CSV 文件路径
    Returns:
        pd.DataFrame: 加载的数据
    """
    print(f"📂 从文件加载: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ 加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
    return df

def get_demo_data(n_samples: int = 200) -> pd.DataFrame:
    """
    获取小批量演示数据，便于快速调试
    Args:
        n_samples: 样本数量
    Returns:
        pd.DataFrame: 小批量数据
    """
    df = load_raw_data()
    return df.head(n_samples).copy()


if __name__ == "__main__":
    # 测试数据加载
    df = load_raw_data()
    print("\n前 5 行数据预览：")
    print(df.head())