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

    # 第一刀：切出 train(70%) 和 rest(30%)
    train_df, rest_df = train_test_split(
        df,
        train_size=0.7,
        stratify=df['label'],       # 分层参数，确保 train 和 rest 中的 spam 占比相同
        random_state=42
    )

    # 第二刀：把 rest 对半切成 val(15%) 和 test(15%)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=0.5,
        stratify=rest_df['label'],  # 分层参数，确保 val 和 test 中的 spam 占比相同
        random_state=42
    )

    # 重置索引，确保从 0 开始
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == '__main__':
    from load import load_sms

    df = load_sms('data/SMSSpamCollection')
    train_df, val_df, test_df = split_data(df)

    # ---- 验收（不要改这部分）----
    total = len(df)
    expected = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    parts = [('train', train_df), ('val', val_df), ('test', test_df)]
    ratios = []
    for name, part in parts:
        ratio = (part['label'] == 'spam').mean()
        ratios.append(ratio)
        share = len(part) / total
        print(f'{name}: {len(part)} 行 ({share:.0%})，spam 占比 {ratio:.1%}')
        assert abs(share - expected[name]) < 0.02, \
            f'{name} 占比 {share:.1%}，期望 {expected[name]:.0%} => 检查 test_size 是相对哪个 DataFrame 算的'
    assert abs(ratios[0] - ratios[1]) < 0.01, 'train 与 val 的 spam 占比差超过 1 个百分点 => 没有分层'
    assert abs(ratios[0] - ratios[2]) < 0.01, 'train 与 test 的 spam 占比差超过 1 个百分点 => 没有分层'
    assert len(train_df) + len(val_df) + len(test_df) == total, '三份行数相加不等于全量 => 有数据丢失或重复'
    print('split.py 验收 PASS')
