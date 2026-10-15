'''
全局配置文件
文本分类的项目的超参数和路径配置
'''
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
print(f'此项目根目录: {PROJECT_ROOT}')


'''
========== 路径配置 ==========
'''
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


'''
========== 数据配置 ==========
'''
DATASET_NAME = '20_newsgroups'
# 本地数据路径（相对于项目根目录）
LOCAL_DATA_PATH = os.path.join(PROJECT_ROOT, '..', '..', '..', 'local_datas', '20_newsgroups')
LOCAL_DATA_PATH = os.path.normpath(LOCAL_DATA_PATH)

# 默认使用4个易区分类别
DEFAULT_CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'rec.sport.baseball'
]

# 目标列
TARGET_COLUMN = 'category'


'''
========== 数据划分 ==========
'''
RANDOM_STATE = 42
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1


'''
========== TF-IDF 配置 ==========
    如果"apple"在一篇文档中出现很多次（TF高），说明这篇文档在讨论苹果
    如果"apple"在很多文档中都出现（IDF低），说明这个词很常见，区分度低
    综合起来，TF-IDF能选出对当前文档重要且在整个数据集中有区分度的词
    
    TF-IDF = 词频(TF) × 逆文档频率(IDF)
    TF-IDF(t, d) = TF(t, d) × IDF(t)

    TF（词频 Term Frequency）:衡量一个词在当前文档中的重要性
    TF(t, d) = 词t在文档d中出现的次数 / 文档d的总词数

    IDF（逆文档频率 Inverse Document Frequency）:衡量一个词的稀有程度，稀有词更有区分度
    IDF(t) = ln( (1+总文档数) / (1+包含词t的文档数) ) + 1
'''
TFIDF_CONFIG = {
    'max_features': 10000,
    'min_df': 2,
    'max_df': 0.8,
    'sublinear_tf': True,
    'stop_words': 'english',
    'ngram_range': (1, 2),
}

'''
========== 模型配置 ==========
'''
MODEL_CONFIGS = {

    # 朴素贝叶斯分类器
    'naive_bayes': {
        'name': 'MultinomialNB',
        'params': {'alpha': 0.1},
    },

    # 逻辑回归分类器
    'logistic': {
        'name': 'LogisticRegression',
        'params': {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': RANDOM_STATE,
        },
    },

    # 支持向量机分类器
    'svm': {
        'name': 'SVC',
        'params': {
            'C': 1.0,
            'kernel': 'linear',
            'max_iter': 10000,
            'probability': True,
            'random_state': RANDOM_STATE,
        },
    },

    # 随机森林分类器
    'forest': {
        'name': 'RandomForestClassifier',
        'params': {
            'n_estimators': 100,
            'max_depth': 20,
            'random_state': RANDOM_STATE,
        },
    },
}

'''
========== 评估配置 ==========
'''
AVERAGE_METHOD = 'macro'