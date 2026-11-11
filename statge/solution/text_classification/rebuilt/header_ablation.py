'''rebuilt 独立重写版：header 剥离实验（量化邮件头的标签泄漏）

背景：
- 20newsgroups 原始帖子的邮件头含 `Newsgroups: alt.atheism` 字段，直接等于标签。
- 项目现在由 configs/text_clean.py 统一剥离 headers/footers/quotes；
  本实验要回答「不剥离会虚高多少」，为文档与面试提供量化依据。

实验设计：
- 同一份数据、同一次划分（random_state=42）、同一 TF-IDF 与 LR 参数，只改文本预处理这一个变量；
- 四个变体逐级剥离，观察每一步的增量影响；
- 只在验证集上比较 macro-F1，测试集全程不动（保持"只用于最终报告"的纪律）；
- 判定口径沿用 experiments.py：Δ 绝对值 >= 0.002 才算有效变化。

运行方式（在 rebuilt 目录内；三端通用，先 conda activate ai_app）：
    python header_ablation.py
'''
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import fit
import load
import split

# 清洗实现只保留 configs/text_clean.py 一份，实验里直接复用，避免规则漂移
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from configs.text_clean import strip_headers, strip_footer, strip_quotes

DELTA_THRESHOLD = 0.002


def build_variants(df):
    '''构造四个文本变体，返回 [(名称, 说明, DataFrame), ...]。'''
    variants = [
        ('V0 原样', '含 headers + footers + quotes（未清洗基线）', df),
    ]

    v1 = df.copy()
    v1['text'] = v1['text'].map(strip_headers)
    variants.append(('V1 去 headers', '仅剥离邮件头', v1))

    v2 = v1.copy()
    v2['text'] = v2['text'].map(strip_footer)
    variants.append(('V2 +去 footers', '剥离邮件头与签名 footer', v2))

    v3 = v2.copy()
    v3['text'] = v3['text'].map(strip_quotes)
    variants.append(('V3 +去 quotes', '三者全剥离（当前生产口径）', v3))

    return variants


def run_one(train_df, val_df):
    '''固定 TF-IDF + LR 参数跑一次，返回 (f1_macro, 空文本数, 平均词数)。'''
    vectorizer = fit.fit_vectorizer(train_df['text'])
    X_train = fit.transform(vectorizer, train_df['text'])
    X_val = fit.transform(vectorizer, val_df['text'])

    model = LogisticRegression(**fit.MODEL_PARAMS)
    model.fit(X_train, train_df['category'])
    y_pred = model.predict(X_val)
    f1 = f1_score(val_df['category'], y_pred, average='macro')

    empty = int((train_df['text'].str.strip() == '').sum())
    avg_words = train_df['text'].str.split().str.len().mean()
    return f1, empty, avg_words


if __name__ == '__main__':
    # 必须取未清洗的原始全文作为 V0 基线，否则对照失去意义
    df = load.load_data(clean=False)

    rows = []
    baseline_f1 = None
    for name, desc, vdf in build_variants(df):
        # 每个变体各自划分：各变体的行序与标签完全一致（只改了 text 列），
        # 因此同 random_state 的分层划分会落在同样的行位置上，样本身份可比。
        # 注意不能先 split 再按 index 回查变体 DataFrame —— split 返回的是
        # reset_index 后的新索引，回查会取错行，甚至让验证集与训练集重叠。
        v_train, v_val, _v_test = split.split_data(vdf)
        f1, empty, avg = run_one(v_train, v_val)
        if baseline_f1 is None:
            baseline_f1 = f1
        rows.append((name, desc, f1, f1 - baseline_f1, empty, avg))
        print(f'{name}: f1_macro={f1:.4f}  空文本={empty}  训练集平均词数={avg:.1f}')

    print()
    print('| 变体 | 处理方式 | 验证集 f1_macro | 与基线差 | 空文本数 | 训练集平均词数 |')
    print('|---|---|---|---|---|---|')
    for name, desc, f1, delta, empty, avg in rows:
        delta_str = '—' if name.startswith('V0') else f'{delta:+.4f}'
        print(f'| {name} | {desc} | {f1:.4f} | {delta_str} | {empty} | {avg:.1f} |')

    print()
    drop = baseline_f1 - rows[-1][2]
    print(f'完全剥离后 F1 下降量: {drop:.4f}（相对下降 {drop / baseline_f1 * 100:.1f}%）')
    if abs(drop) >= 0.05:
        print('结论：邮件头带来的提升显著，未清洗指标虚高，必须披露。')
    elif abs(drop) >= DELTA_THRESHOLD:
        print('结论：邮件头有可测影响但幅度有限，当前已统一剥离，指标可信。')
    else:
        print('结论：邮件头影响低于判定阈值，剥离与否都不改变结论。')
