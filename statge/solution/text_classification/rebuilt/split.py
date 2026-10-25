'''rebuilt 独立重写版：数据集划分（8:1:1 分层）

约束：不得 import 上级 src/ 的任何模块。

运行方式（在 rebuilt 目录内；三端通用，先 conda activate ai_app）：
    python split.py
'''
from sklearn.model_selection import train_test_split


def split_data(df, train_size=0.8, val_size=0.1, random_state=42, label_col='category'):
    '''把 DataFrame 划分为 训练/验证/测试 三份。

    要求：
    1. 两次划分都要 stratify=df[label_col] 分层，保证每类在各集的比例与总体一致
       （不分层时，稀有类别可能被分得比例失衡，指标会失真）。
    2. 固定 random_state，保证三端结果完全一致。
    3. 三份比例为 8:1:1：先切出 20% 的临时集，再把临时集按 val_size/(1-train_size) 对半切。
    4. 返回顺序固定为 (train, val, test)，各自 reset_index(drop=True)。
    5. 测试集在本次训练中只允许用于最终报告，不能参与任何选择（选型只看验证集）。

    参数:
        df:           load_data 返回的 DataFrame
        train_size:   训练集比例
        val_size:     验证集比例
        random_state: 随机种子
        label_col:    用于分层的列名
    返回:
        (train_df, val_df, test_df)
    '''
    train_df, temp_df = train_test_split(
        df, train_size=train_size, test_size=1 - train_size,
        stratify=df[label_col], random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size / (1 - train_size),
        stratify=temp_df[label_col], random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == '__main__':
    from load import load_data

    df = load_data()
    train_df, val_df, test_df = split_data(df)

    # 自测：打印各集 shape、比例，以及每类的占比是否一致
    print(f'训练集: {train_df.shape}  验证集: {val_df.shape}  测试集: {test_df.shape}')
    print(f'实际比例: train={len(train_df) / len(df):.4f} '
          f'val={len(val_df) / len(df):.4f} test={len(test_df) / len(df):.4f}')
    for name, part in (('train', train_df), ('val', val_df), ('test', test_df)):
        print(f'--- {name} 类别分布 ---')
        print(part['category'].value_counts(normalize=True).round(4))
