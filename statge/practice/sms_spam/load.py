"""练习 1/5：数据加载

任务：把 data/SMSSpamCollection（TSV，无表头）读成 DataFrame。

约定：
- 返回 pd.DataFrame，列名 ['label', 'text']
- label ∈ {'ham', 'spam'}
- 加载层不做任何清洗、去重、排序（那是后面步骤的决策）

验收：运行本文件，全部 PASS 才算过。
"""
import pandas as pd


def load_sms(path):
    """读取 TSV，返回 pd.DataFrame，列：['label', 'text']。

    TODO:
      1. 用 pandas 读取（分隔符是什么？没有表头怎么处理？列名在哪传？）
      2. 返回 DataFrame
    """
    raise NotImplementedError('练习：删除本行并实现 load_sms')


if __name__ == '__main__':
    df = load_sms('data/SMSSpamCollection')

    # ---- 验收（不要改这部分）----
    assert isinstance(df, pd.DataFrame), f'返回类型不对: {type(df)}'
    assert list(df.columns) == ['label', 'text'], f'列名不对: {list(df.columns)}'
    assert set(df['label'].unique()) == {'ham', 'spam'}, f'label 取值不对: {df["label"].unique()}'
    assert df['text'].isna().sum() == 0, '存在空 text'
    assert (df['text'].str.strip() == '').sum() == 0, '存在纯空白 text'
    print(f'总行数: {len(df)}（参考: 5574）')
    print(f'spam 占比: {(df["label"] == "spam").mean():.1%}（参考: 约 13.4%）')
    print('load.py 验收 PASS')
