'''
文本分类(Text Classification)
    自然语言处理(NLP)中最基础也是最重要的任务之一。它的目标是将给定的文本文档自动归类到一个或多个预定义的类别中。

    原始文本 => 文本预处理 => 特征提取 => 分类训练 => 分类结果


    文本预处理：
        1. 去除停用词
        2. 词干提取/lemmatization
        3. 标点化
        4. 转换为小写
        5. 词向量表示
        6. 上下文嵌入


    特征提取：
        词袋模型(BoW)
            描述：统计词频
            优点：简单直观
            缺点：忽略词序和语义

        TF-IDF:
            如果"apple"在一篇文档中出现很多次（TF高），说明这篇文档在讨论苹果
            如果""apple"在很多文档中都出现（IDF低），说明这个词很常见，区分度低
            综合起来，TF-IDF能选出对当前文档重要且在整个数据集中有区分度的词

            描述：考虑词的重要性
            优点：比BoW更精确
            缺点：仍然忽略上下文

        Word2Vec
            描述：词向量表示
            优点：捕捉语义关系
            缺点：无法处理多义词

        BERT
            描述：上下文嵌入
            优点：最先进的表示
            缺点：计算资源要求高


    分类模型选择：
        传统机器学习方法：
            朴素贝叶斯
            支持向量机(SVM)
            逻辑回归
            随机森林

        深度学习方法：
            卷积神经网络(CNN)
            循环神经网络(RNN/LSTM)
            Transformer模型(BERT等)
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# -------------------------- 设置中文字体 start --------------------------
plt.rcParams['font.sans-serif'] = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------


'''
1. 加载数据集
'''
# 本地数据集目录
local_data_path = '../local_datas/20_newsgroups'

# 加载数据集
newsgroups = load_files(
    container_path = local_data_path,
    encoding = 'ISO-8859-1',    # 20 Newsgroups 使用这个编码
    shuffle = True,
    random_state = 42,
    decode_error = 'ignore'     # 忽略编码错误
)

# 选择4个类别作为示例(无神论，基督教，计算机图形学，医学科学)
categories = [
    'alt.atheism',
    'soc.religion.christian',
    'comp.graphics',
    'sci.med'
]

print("=" * 50)
print("数据加载完成")
print("=" * 50)
print(f"总样本数: {len(newsgroups.data)}")
print(f"类别名称: {newsgroups.target_names}")
print(f"类别数量: {len(newsgroups.target_names)}")
print("=" * 50)


'''
2. 数据集划分
'''
# 先划分训练集和临时集(验证 + 测试)
X_train, X_temp, y_train, y_temp = train_test_split(
    newsgroups.data,
    newsgroups.target,
    test_size = 0.2,                # 20% 作为验证+测试集
    random_state = 42,
    stratify = newsgroups.target    # 保持类别比例
)

# 再将临时集分为验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size = 0.5,                # 各占一半（即各10%）
    random_state = 42,
    stratify = y_temp
)

print("\n数据划分结果:")
print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本")
print(f"测试集: {len(X_test)} 样本")
print("=" * 50)


'''
3. 特征提取和模型训练
TF-IDF = 词频(TF) × 逆文档频率(IDF)
    TF-IDF(t, d) = TF(t, d) × IDF(t)

TF（词频 Term Frequency）:衡量一个词在当前文档中的重要性
    TF(t, d) = 词t在文档d中出现的次数 / 文档d的总词数

IDF（逆文档频率 Inverse Document Frequency）:衡量一个词的稀有程度，稀有词更有区分度
    IDF(t) = log(总文档数 / 包含词t的文档数 + 1)
'''
# 创建 Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,      # 最大特征数
        stop_words='english',   # 移除英文停用词
        ngram_range=(1, 2),     # 使用一元和二元词袋
        min_df=2,               # 忽略出现次数少于2次的词
        max_df=0.8              # 忽略在80%以上文档中出现的词
    )),
    ('clf', MultinomialNB(alpha=0.1))  # 朴素贝叶斯分类器
])

# 训练模型
print("开始训练模型...")
pipeline.fit(X_train, y_train)
print("训练完成！")
print("=" * 50)


'''
4. 模型评估
'''
# 在验证集上评估
y_val_pred = pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"验证集准确率: {val_accuracy:.4f}")

# 在测试集上评估
y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f}")
print("=" * 50)


'''
5. 详细分类报告
'''
print("\n测试集分类报告:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=newsgroups.target_names
))


'''
6. 混淆矩阵可视化
'''
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=newsgroups.target_names,
    yticklabels=newsgroups.target_names
)
plt.title('混淆矩阵 - 测试集')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.show()


'''
7. 预测示例
'''
print("\n" + "=" * 50)
print("预测示例:")
print("=" * 50)

# 从测试集中取几个样本进行展示
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices[:5]:
    true_label = newsgroups.target_names[y_test[idx]]
    pred_label = newsgroups.target_names[y_test_pred[idx]]
    text_preview = X_test[idx][:200].replace('\n', ' ') + '...'

    print(f"\n文本预览: {text_preview}")
    print(f"真实类别: {true_label}")
    print(f"预测类别: {pred_label}")
    print(f"预测{'正确' if true_label == pred_label else '错误'}")
    print("-" * 50)


# '''
# 8. 保存模型（可选）
# '''
# import joblib
# joblib.dump(pipeline, 'newsgroups_classifier.pkl')
# print("\n模型已保存为 'newsgroups_classifier.pkl'")


# ========== 9. 对比不同模型（扩展）==========
def evaluate_model(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """评估不同分类器的性能"""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', clf)
    ])

    pipeline.fit(X_train, y_train)
    val_acc = pipeline.score(X_val, y_val)
    test_acc = pipeline.score(X_test, y_test)

    return val_acc, test_acc


# 尝试不同模型
print("\n" + "=" * 50)
print("模型对比:")
print("=" * 50)

models = {
    '朴素贝叶斯': MultinomialNB(alpha=0.1),
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '线性SVM': LinearSVC(max_iter=1000, random_state=42)
}

for name, clf in models.items():
    val_acc, test_acc = evaluate_model(
        clf, X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"{name:12} | 验证集: {val_acc:.4f} | 测试集: {test_acc:.4f}")

