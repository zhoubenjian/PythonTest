'''
数据加载模块
从本地 20_newsgroups 目录加载文本数据（使用 sklearn.load_files）
'''
import os
import pandas as pd
from sklearn.datasets import load_files

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    RAW_DATA_DIR, DATASET_NAME, DEFAULT_CATEGORIES,
    RANDOM_STATE, LOCAL_DATA_PATH
)


def load_raw_data(categories=None):
    '''
    从本地目录加载 20_newsgroups 数据集

    参数:
        categories: 指定类别列表，None 则使用默认 4 类

    返回:
        pd.DataFrame: 包含 text 和 category 两列的 DataFrame
    '''
    if not os.path.exists(LOCAL_DATA_PATH):
        raise FileNotFoundError(
            f"本地数据目录不存在: {LOCAL_DATA_PATH}\n"
            f"请确认数据已放置在正确位置"
        )

    print(f"📂 从本地加载数据...")
    print(f"   路径: {LOCAL_DATA_PATH}")

    data = load_files(
        container_path=LOCAL_DATA_PATH,
        encoding='ISO-8859-1',
        shuffle=True,
        random_state=RANDOM_STATE,
        decode_error='ignore',
    )

    df = pd.DataFrame({
        'text': data.data,
        'category_id': data.target,
    })
    df['category'] = df['category_id'].map(lambda x: data.target_names[x])

    if categories is not None:
        df = df[df['category'].isin(categories)].reset_index(drop=True)
        print(f"   已过滤为 {len(categories)} 个类别: {categories}")

    print(f"✅ 数据加载完成: {len(df)} 条样本")
    print(f"   类别映射: {dict(enumerate(df['category'].unique()))}")

    return df


def save_raw_data(df, filename=None):
    '''
    将原始数据保存为 CSV
    '''
    if filename is None:
        filename = f"{DATASET_NAME}_raw.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"💾 原始数据已保存: {filepath}")
    return filepath


def load_from_csv(filepath):
    '''
    从本地 CSV 加载数据
    '''
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"📂 从 CSV 加载数据: {filepath} ({len(df)} 条)")
    return df


def get_demo_data(n=5):
    '''
    获取演示用的小批量数据
    '''
    df = load_raw_data()
    return df.head(n)


if __name__ == '__main__':
    df = load_raw_data(categories=DEFAULT_CATEGORIES)
    save_raw_data(df)
    print("\n类别分布:")
    print(df['category'].value_counts())