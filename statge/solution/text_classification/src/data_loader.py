'''
数据加载模块
策略：
  1. sklearn 缓存就绪 → fetch_20newsgroups（秒走缓存，三端通用）
  2. sklearn 缓存未就绪 + 项目内 local_datas 存在 → load_files（本机兜底）
  3. 都没有 → 报错提示联网 fetch
'''
import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups, load_files, get_data_home

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    RAW_DATA_DIR, DATASET_NAME, DEFAULT_CATEGORIES,
    RANDOM_STATE,
)


def _sklearn_cache_ready():
    """检查 sklearn 20newsgroups 缓存是否就绪。"""
    cache_root = os.path.join(get_data_home(), '20news_home', '20news-bydate')
    train_dir = os.path.join(cache_root, '20news-bydate-train')
    if not os.path.isdir(train_dir):
        return False
    subdirs = [d for d in os.listdir(train_dir)
              if os.path.isdir(os.path.join(train_dir, d))]
    return len(subdirs) > 0


def _local_fallback_root():
    """项目内 local_datas/20_newsgroups 路径。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, '..', '..', '..', '..', 'local_datas', '20_newsgroups')
    )


def load_raw_data(categories=None):
    '''
    加载 20_newsgroups 数据集

    参数:
        categories: 指定类别列表，None 则使用默认 4 类
    返回:
        pd.DataFrame: 包含 text 和 category 两列的 DataFrame
    '''
    cats = categories if categories is not None else DEFAULT_CATEGORIES
    local_root = _local_fallback_root()

    if _sklearn_cache_ready():
        print(f"📂 sklearn 缓存已就绪 → fetch_20newsgroups")
        data = fetch_20newsgroups(
            subset='all',
            categories=cats,
            shuffle=True,
            random_state=RANDOM_STATE,
            remove=('headers', 'footers', 'quotes'),
        )
    elif os.path.isdir(local_root):
        print(f"📂 sklearn 缓存未就绪 → fallback 本地 load_files")
        print(f"   路径: {local_root}")
        data = load_files(
            container_path=local_root,
            categories=cats,
            encoding='ISO-8859-1',
            shuffle=True,
            random_state=RANDOM_STATE,
            decode_error='ignore',
        )
    else:
        raise RuntimeError(
            "找不到 20newsgroups 数据。\n"
            "  请联网执行：python -c \"from sklearn.datasets import fetch_20newsgroups; fetch_20newsgroups(subset='train')\""
        )

    df = pd.DataFrame({
        'text': data.data,
        'category_id': data.target,
    })
    df['category'] = df['category_id'].map(lambda x: data.target_names[x])

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
