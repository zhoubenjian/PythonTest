"""练习 2/5：数据划分

任务：把全量数据分层划分成 train 70% / val 15% / test 15%。

约定：
- random_state=42，保证可复现
- 必须分层（stratify）——做完要能回答「为什么必须分层」

验收：运行本文件，三份的 spam 占比差 < 1 个百分点。
"""
from sklearn.model_selection import train_test_split


def split_data(df, random_state=42):
    """分层划分，返回 (train_df, val_df, test_df)。

    TODO:
      1. 第一刀：切出 train(70%) 和 rest(30%)
      2. 第二刀：把 rest 对半切成 val 和 test
      3. 返回三个 DataFrame

    提醒：分层参数传的是哪一列？两刀都要传吗？
    """
    raise NotImplementedError('练习：删除本行并实现 split_data')


if __name__ == '__main__':
    from load import load_sms

    df = load_sms('data/SMSSpamCollection')
    train_df, val_df, test_df = split_data(df)

    # ---- 验收（不要改这部分）----
    total = len(df)
    parts = [('train', train_df), ('val', val_df), ('test', test_df)]
    ratios = []
    for name, part in parts:
        ratio = (part['label'] == 'spam').mean()
        ratios.append(ratio)
        print(f'{name}: {len(part)} 行 ({len(part) / total:.0%})，spam 占比 {ratio:.1%}')
    assert abs(ratios[0] - ratios[1]) < 0.01, 'train 与 val 的 spam 占比差超过 1 个百分点 → 没有分层'
    assert abs(ratios[0] - ratios[2]) < 0.01, 'train 与 test 的 spam 占比差超过 1 个百分点 → 没有分层'
    print('split.py 验收 PASS')
