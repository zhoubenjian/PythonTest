'''rebuilt 独立重写版：数据加载

约束：
1. 本目录下的代码不得 import 上级 src/ 的任何模块，也不得复用其中的函数。
2. 实现由自己完成，AI 只做审查。已给出的只有接口签名与要求说明。

运行方式（在 rebuilt 目录内，用 ai_app 环境；三端通用，先 conda activate ai_app）：
    python load.py
'''
import os

import pandas as pd
from sklearn.datasets import fetch_20newsgroups, load_files, get_data_home

DEFAULT_CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'rec.sport.baseball',
]

RANDOM_STATE = 42


def _sklearn_cache_ready():
    """检查 sklearn 20newsgroups 缓存是否已就绪（避免联网超时）。"""
    cache_root = os.path.join(get_data_home(), '20news_home', '20news-bydate')
    train_dir = os.path.join(cache_root, '20news-bydate-train')
    if not os.path.isdir(train_dir):
        return False
    # 至少要有一个类别子目录
    subdirs = [d for d in os.listdir(train_dir)
              if os.path.isdir(os.path.join(train_dir, d))]
    return len(subdirs) > 0


def _local_fallback_root():
    """项目内 local_datas/20_newsgroups 路径（本机兜底）。"""
    _HERE = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(
        os.path.join(_HERE, '..', '..', '..', '..', 'local_datas', '20_newsgroups')
    )
    return path


def load_data(categories=None, random_state=RANDOM_STATE):
    '''加载 20newsgroups 的指定类别，返回 DataFrame。

    策略：
      1. sklearn 缓存已就绪 → fetch_20newsgroups（秒走缓存）
      2. sklearn 缓存未就绪 + 项目内 local_datas 存在 → load_files（本机兜底）
      3. 都没有 → 报错，提示联网 fetch

    此逻辑避免在网络不通时 fetch 卡 120 秒超时。
    Linux/Mac 首次运行前请先联网执行一次 fetch_20newsgroups 建立缓存。

    参数:
        categories:   要保留的类别名列表，None 表示用默认 4 类
        random_state: 随机种子
    返回:
        pd.DataFrame
    '''
    cats = categories if categories is not None else DEFAULT_CATEGORIES
    local_root = _local_fallback_root()

    if _sklearn_cache_ready():
        # 缓存就绪 → fetch（秒返回）
        data = fetch_20newsgroups(
            subset='all',
            categories=cats,
            shuffle=True,
            random_state=random_state,
            remove=('headers', 'footers', 'quotes'),
        )
    elif os.path.isdir(local_root):
        # 缓存未就绪但项目内有本地数据 → load_files（本机兜底）
        data = load_files(
            container_path=local_root,
            categories=cats,
            encoding='ISO-8859-1',
            shuffle=True,
            random_state=random_state,
            decode_error='ignore',
        )
    else:
        raise RuntimeError(
            "找不到 20newsgroups 数据。\n"
            "  1. 联网执行一次：python -c \"from sklearn.datasets import fetch_20newsgroups; fetch_20newsgroups(subset='train')\"\n"
            "  2. 或手动将数据放到 sklearn 缓存目录。"
        )

    df = pd.DataFrame({
        'text': data.data,
        'category_id': data.target
    })
    df['category'] = df['category_id'].map(lambda x: data.target_names[x])

    return df.reset_index(drop=True)


if __name__ == '__main__':
    # 自测入口：跑通后应能打印总样本数与每类样本数
    df = load_data()
    print(f'总样本数: {len(df)}')
    print(df['category'].value_counts())
    print(df.head(2))
