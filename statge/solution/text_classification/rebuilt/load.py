'''rebuilt 独立重写版：数据加载

约束：
1. 本目录下的代码不得 import 上级 src/ 的任何模块，也不得复用其中的函数。
2. 实现由自己完成，AI 只做审查。已给出的只有接口签名与要求说明。

运行方式（在 rebuilt 目录内，用 ai_app 环境）：
    & "D:\\ProgramData\\anaconda3\\envs\\ai_app\\python.exe" load.py
'''
import os

import pandas as pd
from sklearn.datasets import load_files

# 数据目录：仓库根 / local_datas / 20_newsgroups
# 层级：rebuilt -> text_classification -> solution -> statge -> 仓库根（4 级）
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(
    os.path.join(_HERE, '..', '..', '..', '..', 'local_datas', '20_newsgroups')
)

DEFAULT_CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'rec.sport.baseball',
]

RANDOM_STATE = 42


def load_data(categories=None, data_root=DATA_ROOT, random_state=RANDOM_STATE):
    '''加载 20newsgroups 的指定类别，返回 DataFrame。

    要求：
    1. 数据目录不存在时抛 FileNotFoundError，错误信息里带上实际路径。
    2. 用 sklearn.datasets.load_files 读取；encoding='ISO-8859-1'、decode_error='ignore'
       （这批数据是 1980 年代末的邮件，含大量非 UTF-8 字节，用默认 UTF-8 会直接报错）。
    3. shuffle=True 且固定 random_state —— 三端（Windows/Ubuntu/Mac）必须得到完全相同的行顺序。
    4. categories 为 None 时使用 DEFAULT_CATEGORIES，否则只保留传入的类别。
    5. category 列由 target_names 映射得到（不要用 unique() 的出现顺序猜类别编号）。
    6. 返回的 DataFrame 至少包含 text / category 两列，index 重置为 0..n-1。

    参数:
        categories:   要保留的类别名列表，None 表示用默认 4 类
        data_root:    数据目录，默认取 DATA_ROOT
        random_state: 随机种子
    返回:
        pd.DataFrame
    '''
    raise NotImplementedError('TODO: 按 docstring 里的 6 条要求实现')


if __name__ == '__main__':
    # 自测入口：跑通后应能打印总样本数与每类样本数（本版本 4 类共 4000 条）
    df = load_data()
    print(f'总样本数: {len(df)}')
    print(df['category'].value_counts())
    print(df.head(2))
