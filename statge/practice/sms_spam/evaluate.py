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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate(test_df, model, vectorizer):
    """在 test 上评估，打印指标，返回 (accuracy, f1_macro)。

    TODO:
      1. transform(test_df['text']) => predict
      2. 打印 accuracy、macro-F1、混淆矩阵（标明行/列顺序是 [ham, spam]）
      3. 打印 3 条错分示例：原文 + 真实标签 + 预测标签
      4. 返回 (accuracy, f1)
    """

    from fit import transform

    X_test = transform(vectorizer, test_df['text'])
    y_test = test_df['label'].to_numpy()
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'test accuracy = {accuracy:.4f}\n')

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f'test macro-F1 = {f1:.4f}\n')

    cm = confusion_matrix(y_test, y_pred)
    print('混淆矩阵（行=真实，列=预测，顺序 [ham, spam]）:')
    print(cm, '\n')

    # 打印 3 条错分示例（真正错分的，不是前 3 条）
    import numpy as np

    wrong_idx = np.flatnonzero(y_test != y_pred)
    texts = test_df['text'].to_numpy()
    print(f'错分 {len(wrong_idx)} / {len(y_test)} 条，示例前 3 条：')
    for i in wrong_idx[:3]:
        print(f'原文: {texts[i][:100]}')
        print(f'真实标签: {y_test[i]}  预测标签: {y_pred[i]}')
        print('-' * 50)

    return accuracy, f1


if __name__ == '__main__':
    import os

    from sklearn.metrics import f1_score

    from fit import fit_vectorizer, transform, train_and_compare
    from load import load_sms
    from split import split_data

    df = load_sms('data/SMSSpamCollection')
    train_df, val_df, test_df = split_data(df)

    best_model, best_name, vectorizer = train_and_compare(train_df, val_df)
    # val_f1 选择「在这里重算」而不是改 train_and_compare 的返回签名：
    # 改签名会连带改 fit.py 的验收块，而 val 本来就是选型用的、可以反复评估，
    # 重算一行更简单，也不污染 fit.py 的接口。
    X_val = transform(vectorizer, val_df['text'])
    y_val = val_df['label'].to_numpy()
    val_f1 = f1_score(y_val, best_model.predict(X_val), average='macro')

    acc, test_f1 = evaluate(test_df, best_model, vectorizer)

    # ---- 验收（不要改这部分）----
    print(f'test accuracy={acc:.4f}  test macro-F1={test_f1:.4f}  (best={best_name})')
    assert abs(test_f1 - val_f1) < 0.05, f'test 与 val 差 {abs(test_f1 - val_f1):.4f} >= 0.05，怀疑选型泄漏或数据异常'
    print(f'evaluate.py 验收 PASS（val {val_f1:.4f} / test {test_f1:.4f}）')

    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(best_model, 'artifacts/model.pkl')
    joblib.dump(vectorizer, 'artifacts/vectorizer.pkl')
    print('已落盘 artifacts/model.pkl、artifacts/vectorizer.pkl（练习 5 用）')
