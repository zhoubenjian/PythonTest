'''rebuilt 独立重写版：W02 三组受控实验（ngram / 模型对比 / 去重前后）

约束：
1. 复用同目录 load / split / fit 模块的函数；不得 import 上级 src/。
2. 实现由自己完成，AI 只做审查。

实验设计（执行日志 W02 #2 已定案，不要改动设计只做实现）：
- 三组共用同一次 load_data + split_data（random_state 不变），只动一个变量；
- 判定标准：验证集 macro-F1 变化 >= 0.002 才算有效（1 条样本 = 0.0025）；
- 实验一 ngram：TF-IDF (1,1) vs (1,2)，模型固定 LogisticRegression 默认；
- 实验二 模型对比：LR / LinearSVC / MultinomialNB，TF-IDF 固定 (1,1)；
  注意 LinearSVC 没有 predict_proba，本实验只比 f1_macro，不比置信度；
- 实验三 去重前后：原始数据 vs 按 text 全库去重（每对 cross-post 留 1 条，
  决策 A），划分参数不变；基线复用原始数据上的 LR (1,1) 结果，不重复跑。

输出：一张 markdown 实验记录表（实验 / 变量 / 验证集 f1_macro / 与基线差 / 结论），
print 出来直接贴进执行日志，即为 W02 验收的"实验记录表"。

运行方式（在 rebuilt 目录内；三端通用，先 conda activate ai_app）：
    python experiments.py
'''
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import fit
import load
import split

# 与 fit.py 的默认参数保持同一份口径；ngram 实验只覆盖 ngram_range 这一个键
BASE_TFIDF_PARAMS = dict(fit.TFIDF_PARAMS)
DELTA_THRESHOLD = 0.002  # 判定有效提升的最小幅度


def dedup_crossposts(df):
    '''按 text 全库去重，保留每对重复中的第一条，返回新 DataFrame。

    要求：
    1. 打印去重前后条数（预期 4000 -> 3995，若不是请核对你的数据）；
    2. 不修改传入的 df（去重实验的对照还要用原始数据）。
    '''
    raise NotImplementedError


def run_one(train_df, val_df, tfidf_params, model):
    '''跑一次"向量化 -> 训练 -> 验证集评估"，返回 f1_macro（float）。

    要求：
    1. 只用 fit.fit_vectorizer / fit.transform / fit.evaluate；
    2. model 是已构造好的实例（本函数不负责选择模型种类）；
    3. 不要打印 classification_report（实验表只要数字，报告噪音太大）。
    '''
    raise NotImplementedError


def experiment_ngram(train_df, val_df):
    '''实验一：ngram (1,1) vs (1,2)。返回 [(名称, f1), ...] 两条记录。

    要求：模型统一 LogisticRegression 默认（fit.MODEL_PARAMS）。
    '''
    raise NotImplementedError


def experiment_models(train_df, val_df):
    '''实验二：LR / LinearSVC / MultinomialNB 三模型对比。返回 [(名称, f1), ...] 三条。

    要求：
    1. TF-IDF 固定 BASE_TFIDF_PARAMS；
    2. LR 直接用 fit.train_model（等价于 LogisticRegression(**fit.MODEL_PARAMS)），
       另两个模型在这里自行构造，不要改 fit.py；
    3. 三者超参一律默认，保证"只动模型种类"这一个变量。
    '''
    raise NotImplementedError


def experiment_dedup(train_df, val_df, baseline_f1):
    '''实验三：去重前后对照。返回 [(名称, f1), ...] 一条（去重后）。

    要求：
    1. baseline_f1 是原始数据上的 LR (1,1) 结果，这里只跑去重后那次；
    2. 对 train_df / val_df 分别 dedup_crossposts 后再跑（划分参数不变），
       并打印去重后 train/val 的条数与各类占比变化。
    '''
    raise NotImplementedError


def print_table(rows, baseline_f1):
    '''把所有实验记录打印成 markdown 表格。

    要求：列 = 实验 / 变量 / 验证集 f1_macro / 与基线差 / 是否有效(>=0.002)。
    与基线差保留 4 位小数；基线本身那行"与基线差"写 0.0000。
    '''
    raise NotImplementedError


if __name__ == '__main__':
    # 共用同一次加载与划分：三组实验的固定量，任何一组都不得重新 split
    df = load.load_data()
    train_df, val_df, _test_df = split.split_data(df)

    # 基线：原始数据 + LR 默认 + TF-IDF (1,1)，预期 0.9950
    baseline_f1 = run_one(train_df, val_df, BASE_TFIDF_PARAMS,
                          LogisticRegression(**fit.MODEL_PARAMS))
    print(f'基线 f1_macro = {baseline_f1:.4f}\n')

    rows = [('基线', 'LR + (1,1) + 原始数据', baseline_f1)]
    rows += experiment_ngram(train_df, val_df)
    rows += experiment_models(train_df, val_df)
    rows += experiment_dedup(train_df, val_df, baseline_f1)
    print_table(rows, baseline_f1)
