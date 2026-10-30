# -*- coding: utf-8 -*-
'''W02 任务 #3：错分样本库（方案 A：5 折 OOF 错分 + 低置信度近错样本）。

背景：验证集 400 条只错 ~2 条，凑不齐 30 条分析目标。方案 A 用 5 折分层
交叉验证让每个样本都拿到"没见过它的模型"的预测（out-of-fold, OOF），
全部错分都是泛化误差；再补置信度最低的近错样本（答对但很悬），
合计 ~40 条进入人工归因。

人工归因标签（实现完跑出样本表后，逐条读 text 打在 error_type 列）：
    cross_post       重复/近重复残留（对照实验三）
    domain_confusion 域混淆（如 space 与 med 的交叉话题）
    noise_short      短文本 / 噪声 / 编码损坏
    label_suspect    标注可疑（模型可能是对的）
归因产物四件套：归因分布表、混淆矩阵、per-class 召回、3~5 个代表案例。

约定与 experiments.py 一致：
- 只 import rebuilt 内的 load/fit，不 import src/*
- 全程使用 fit.MODEL_PARAMS 与 BASE_TFIDF_PARAMS（从 fit / experiments 引入），
  保证与实验基线同一配置
- 随机种子统一 42
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import fit
from load import load_data

N_SPLITS = 5          # 5 折分层交叉验证
NEAR_MISS_TOP = 20    # 近错样本条数：答对但置信度最低的 Top-N


def oof_predictions(df, n_splits=N_SPLITS):
    '''对全量数据做 5 折分层 OOF 预测。

    要求：
    1. StratifiedKFold(n_splits, shuffle=True, random_state=42)
    2. 逐折：fit.fit_vectorizer(训练折) -> transform 两折 ->
       LogisticRegression(**fit.MODEL_PARAMS) 拟合 -> 预测验证折
       （严禁在全量上先 fit vectorizer 再进折内 —— 那是泄漏）
    3. 每条样本记录: text / y_true(真实类别字符串) / y_pred /
       correct(bool) / confidence(predict_proba 的最大值)
    4. 返回 (df_oof, fold_f1s)：df_oof 行数 == len(df)；
       fold_f1s 是逐折 f1_macro 列表（长度 n_splits）
    预期：~4000 条 → 错分约 15~25 条（f1≈0.995 对应），若明显偏离先自查折内 fit
    '''

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df_oofs, fold_f1s = [], []

    for train_idx, val_idx in skf.split(df, df['category']):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # 1. 只在训练折上 fit vectorizer，并保存返回值
        vectorizer = fit.fit_vectorizer(train_df['text'])

        # 2. 分别 transform 训练折和验证折
        X_train = fit.transform(vectorizer, train_df['text'])
        X_val = fit.transform(vectorizer, val_df['text'])

        # 3. 用训练折训练模型（用 fit.train_model 或直接 LogisticRegression）
        model = fit.train_model(X_train, train_df['category'])

        # 4. 预测验证折
        y_pred = model.predict(X_val)
        proba = model.predict_proba(X_val)
        confidence = np.max(proba, axis=1)

        # 5. 存这一折的 f1
        fold_f1s.append(f1_score(val_df['category'], y_pred, average='macro'))

        # 6. 存这一折的预测结果
        df_oofs.append(pd.DataFrame({
            'text': val_df['text'],
            'y_true': val_df['category'],
            'y_pred': y_pred,
            'correct': (y_pred == val_df['category']),
            'confidence': confidence
        }))

    # 7. 合并所有折的预测结果
    df_oof = pd.concat(df_oofs, ignore_index=True)
    return df_oof, fold_f1s


def collect_misclassified_and_nearmiss(df_oof, top_n=NEAR_MISS_TOP):
    '''从 OOF 结果里拆出错分样本表与近错样本表。

    要求：
    1. 错分表: correct == False 的全部行，按 confidence 升序（错得最狠的排最前）
    2. 近错表: correct == True 里 confidence 最低的 top_n 行，同样升序
    3. 都不修改 df_oof 本身
    4. 返回 (df_mis, df_near)
    '''

    # 1. 错分表：correct == False，confidence 升序
    df_mis = df_oof[df_oof['correct'] == False].copy().sort_values(by='confidence', ascending=True)

    # 2. 近错表：correct == True，confidence 升序，取前 top_n
    df_near = df_oof[df_oof['correct'] == True].copy().sort_values(by='confidence', ascending=True).head(top_n)

    # 3. 返回(重置索引)
    return df_mis.reset_index(drop=True), df_near.reset_index(drop=True)


def confusion_and_per_class(df_oof):
    '''打印混淆矩阵（带行列类别标签）与 per-class 分类报告。

    要求：
    1. labels 用 sorted(df_oof['y_true'].unique())，矩阵行=真实、列=预测
    2. 先打矩阵（含标签表头），再打 classification_report
    3. 无返回值，只打印
    '''

    labels = sorted(df_oof['y_true'].unique())

    conf_mat = confusion_matrix(df_oof['y_true'], df_oof['y_pred'], labels=labels)

    print(pd.DataFrame(conf_mat, index=labels, columns=labels).to_string())
    print('\n' + '=' * 50)
    print(classification_report(df_oof['y_true'], df_oof['y_pred'], labels=labels, target_names=labels))


if __name__ == '__main__':
    # 注意：这里不做 split_data —— OOF 自己就是划分机制；也不去重，
    # 保持与最终模型一致的原始数据分布（cross_post 是归因标签之一）
    df = load_data()
    df_oof, fold_f1s = oof_predictions(df)

    print('逐折 f1_macro:', [f'{x:.4f}' for x in fold_f1s])
    print(f'CV f1_macro = {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}')

    df_mis, df_near = collect_misclassified_and_nearmiss(df_oof)
    print(f'\n错分 {len(df_mis)} 条（最错的在前）/ 近错 {len(df_near)} 条（最悬的在前）')

    for name, sub in [('错分', df_mis), ('近错', df_near)]:
        print(f'\n===== {name}样本预览（text 截断到 80 字符）=====')
        preview = sub.copy()
        preview['text'] = preview['text'].str[:80].str.replace('\n', ' ')
        print(preview[['y_true', 'y_pred', 'confidence', 'text']].to_string())

    print('\n===== 混淆矩阵与 per-class 报告 =====')
    confusion_and_per_class(df_oof)

    # 下一步（人工）：把上面两张表导出/复制，逐条读 text 加 error_type 列，
    # 汇总归因分布表写进 执行日志.md 的 W02 记录
