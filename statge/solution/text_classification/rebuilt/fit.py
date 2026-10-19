'''rebuilt 独立重写版：TF-IDF 特征提取 + 训练 + 验证集评估

约束：不得 import 上级 src/ 的任何模块。

W01 只做单模型（LogisticRegression）最小闭环，目的是验证自己理解了流程，
不是在重做四模型对比（那已经在基线 w01_baseline_test.log 里有了）。
多模型对比、选型口径统一留给 W02。

运行方式（在 rebuilt 目录内）：
    & "D:\\ProgramData\\anaconda3\\envs\\ai_app\\python.exe" fit.py
'''
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
# 产物写在 rebuilt 自己的目录里，不覆盖上级 models/ 下由旧环境（sklearn 1.6.1）生成的 pkl
ARTIFACT_DIR = os.path.join(_HERE, 'artifacts')
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pkl')

# 与原 configs/config.py 保持一致的 TF-IDF 配置；是否认同这些取值写进 NOTES.md
TFIDF_PARAMS = {
    'max_features': 10000,
    'min_df': 2,
    'max_df': 0.8,
    'sublinear_tf': True,
    'stop_words': 'english',
    'ngram_range': (1, 2),
}

MODEL_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'solver': 'lbfgs',
    'random_state': 42,
}


def fit_vectorizer(texts_train, tfidf_params=None):
    '''只在训练集文本上拟合 TF-IDF，返回 TfidfVectorizer。

    要求：
    1. 只允许用训练集 fit —— 验证集和测试集的词汇信息一旦进入向量化器，
       就是数据泄漏（IDF、词表都会被污染），指标会虚高。
    2. 参数默认取 TFIDF_PARAMS。

    参数:
        texts_train: 训练集文本（可迭代）
        tfidf_params: 覆盖默认参数，None 表示用 TFIDF_PARAMS
    返回:
        TfidfVectorizer（已 fit）
    '''
    raise NotImplementedError('TODO: 实现，注意只用训练集 fit')


def transform(vectorizer, texts):
    '''用已拟合的 vectorizer 把文本转成稀疏矩阵（只 transform，不 fit）。'''
    raise NotImplementedError('TODO: 实现')


def train_model(X_train, y_train, model_params=None):
    '''训练 LogisticRegression。

    说明：不引入 LabelEncoder，直接把字符串标签交给 sklearn
    （LogisticRegression 支持字符串 y），少一层编解码就少一个出错点。
    '''
    raise NotImplementedError('TODO: 实现')


def evaluate(model, X, y_true):
    '''在给定数据集上评估，返回 dict 并打印分类报告。

    返回的 dict 至少包含:
        accuracy: 准确率
        f1_macro: macro 平均 F1（4 类均衡，用 macro 而不是 weighted）
    '''
    raise NotImplementedError('TODO: 实现')


def save_artifacts(vectorizer, model):
    '''把 vectorizer 与 model 分别存到 ARTIFACT_DIR，供 predict.py 在新进程加载。

    要求：目录不存在时先创建。
    '''
    raise NotImplementedError('TODO: 实现，用 joblib.dump')


if __name__ == '__main__':
    from load import load_data
    from split import split_data

    df = load_data()
    train_df, val_df, test_df = split_data(df)

    # 1) 只在训练集上拟合向量化器，再分别 transform
    vectorizer = fit_vectorizer(train_df['text'])
    X_train = transform(vectorizer, train_df['text'])
    X_val = transform(vectorizer, val_df['text'])
    print(f'训练集矩阵: {X_train.shape}  验证集矩阵: {X_val.shape}')

    # 2) 训练
    model = train_model(X_train, train_df['category'])

    # 3) 只看验证集
    val_metrics = evaluate(model, X_val, val_df['category'])
    print(f"验证集 macro-F1: {val_metrics['f1_macro']:.4f}")

    # 4) 保存产物，供 predict.py 使用
    save_artifacts(vectorizer, model)

    # TODO(W02): 需不需要在这里用测试集出最终报告？注意测试集只能报告、不能参与选择
