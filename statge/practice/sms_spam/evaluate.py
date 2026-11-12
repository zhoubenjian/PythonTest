"""练习 4/5：测试集终评

任务：用 fit.py 选出的 best 模型在 test 上跑一次完整评估。

红线：
- 测试集只能跑一次。不许根据 test 分数回头改超参再跑——那是测试集泄漏，
  改了就得声明换数据重来。
- y 用 1=spam / 0=ham（与 fit.py 的验收保持一致）

产出（顺手完成，练习 5 要用）：
- 把 best_model 和 vectorizer joblib.dump 到 artifacts/model.pkl、artifacts/vectorizer.pkl

验收：运行本文件，test F1 与 val F1 差 < 0.05，且错分示例已打印。
"""
import joblib


def evaluate(test_df, model, vectorizer):
    """在 test 上评估，打印指标，返回 (accuracy, f1_macro)。

    TODO:
      1. transform(test_df['text']) → predict
      2. 打印 accuracy、macro-F1、混淆矩阵（标明行/列顺序是 [ham, spam]）
      3. 打印 3 条错分示例：原文 + 真实标签 + 预测标签
      4. 返回 (accuracy, f1)
    """
    raise NotImplementedError('练习：删除本行并实现 evaluate')


if __name__ == '__main__':
    import os

    from sklearn.metrics import f1_score

    from fit import fit_vectorizer, transform, train_and_compare
    from load import load_sms
    from split import split_data

    df = load_sms('data/SMSSpamCollection')
    train_df, val_df, test_df = split_data(df)

    best_model, best_name, vectorizer = train_and_compare(train_df, val_df)
    # TODO: val_f1 从哪来？train_and_compare 里算过但没返回。
    # 自己决定改法：改返回签名，还是在这里重算。两种都对，想清楚你选哪个、为什么。

    acc, test_f1 = evaluate(test_df, best_model, vectorizer)

    # ---- 验收（不要改这部分）----
    print(f'test accuracy={acc:.4f}  test macro-F1={test_f1:.4f}  (best={best_name})')
    # TODO: 填上 val_f1 后启用下面两行
    # assert abs(test_f1 - val_f1) < 0.05, f'test 与 val 差 {abs(test_f1 - val_f1):.4f} >= 0.05，怀疑选型泄漏或数据异常'
    # print('evaluate.py 验收 PASS')

    os.makedirs('artifacts', exist_ok=True)
    # TODO: joblib.dump(best_model, 'artifacts/model.pkl')
    # TODO: joblib.dump(vectorizer, 'artifacts/vectorizer.pkl')
    print('（落盘 artifacts/ 是练习 5 的前置，别忘）')
