"""练习 3/5：特征工程 + 模型选型

任务：TF-IDF 向量化，两个模型在 val 上对比，选最好的。

约定：
- TF-IDF max_features=5000
- 只在训练集上 fit 向量化器（做完要能回答：在全量上 fit 是什么泄漏？）
- 两个模型自选（建议 MultinomialNB + LogisticRegression，换别的也行）
- best 按 val 分数选，不许写死模型名
- y 编码约定：1=spam / 0=ham（本文件验收块和练习 4 都按这个口径写）

验收：运行本文件，最佳 val macro-F1 >= 0.85。
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def fit_vectorizer(train_texts):
    """只在训练集上 fit，返回 TfidfVectorizer。

    TODO:
      1. 创建 TfidfVectorizer(max_features=5000)
      2. fit_transform(train_texts)，返回 vectorizer（转换结果在调用方算）
    """

    vectorizer = TfidfVectorizer(max_features=5000)
    # 只在训练集上 fit，避免验证集/测试集的词表泄漏到训练阶段
    vectorizer.fit(train_texts)
    return vectorizer


def transform(vectorizer, texts):
    """用已有 vectorizer 转换任意文本集。"""

    return vectorizer.transform(texts)


def train_and_compare(train_df, val_df):
    """训练两个模型，在 val 上比 macro-F1。

    返回 (best_model, best_name, vectorizer)。

    TODO:
      1. fit_vectorizer(train_df['text'])
      2. 训练两个模型
      3. 各自算 val macro-F1 并打印
      4. 按分数选 best 返回
    """

    # 1. 只在训练集上 fit 向量化器，避免 val/test 词表泄漏
    vectorizer = fit_vectorizer(train_df['text'])
    X_train = transform(vectorizer, train_df['text'])
    X_val = transform(vectorizer, val_df['text'])
    # 0:ham（正常短信），1:spam（垃圾短信）
    y_train = (train_df['label'] == 'spam').astype(int)
    y_val = (val_df['label'] == 'spam').astype(int)

    # 2. 训练两个模型并各自评估
    mnb_model = MultinomialNB()
    mnb_model.fit(X_train, y_train)
    mnb_f1 = f1_score(y_val, mnb_model.predict(X_val), average='macro')

    lr_model = LogisticRegression(max_iter=5000)
    lr_model.fit(X_train, y_train)
    lr_f1 = f1_score(y_val, lr_model.predict(X_val), average='macro')

    print(f'MultinomialNB（朴素贝叶斯） val macro-F1 = {mnb_f1:.4f}')
    print(f'LogisticRegression（逻辑回归） val macro-F1 = {lr_f1:.4f}')

    # 3. 按 val 分数选 best（不许写死模型名）
    scores = {
        'MultinomialNB（朴素贝叶斯）': (mnb_f1, mnb_model), 
        'LogisticRegression（逻辑回归）': (lr_f1, lr_model)
    }
    best_name, (best_f1, best_model) = max(scores.items(), key=lambda kv: kv[1][0])
    # print(f'best = {best_name}（val marco-F1 = {best_f1:.4f}）')

    return best_model, best_name, vectorizer


if __name__ == '__main__':
    from load import load_sms
    from split import split_data

    df = load_sms('data/SMSSpamCollection')
    train_df, val_df, _test_df = split_data(df)

    best_model, best_name, vectorizer = train_and_compare(train_df, val_df)

    # ---- 验收（不要改这部分）----
    X_val = transform(vectorizer, val_df['text'])
    y_val = (val_df['label'] == 'spam').astype(int)
    f1 = f1_score(y_val, best_model.predict(X_val), average='macro')
    print(f'best = {best_name}, val macro-F1 = {f1:.4f}')
    assert f1 >= 0.85, f'val macro-F1 {f1:.4f} < 0.85，先查选型逻辑再想特征'
    print('fit.py 验收 PASS')
