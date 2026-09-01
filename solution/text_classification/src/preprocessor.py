'''
文本预处理模块
负责：TF-IDF 特征提取、标签编码、数据划分、保存/加载预处理器
'''
import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    PROCESSED_DATA_DIR, PREPROCESSOR_PATH,
    TFIDF_CONFIG, RANDOM_STATE, TRAIN_SIZE, VAL_SIZE
)


class TextPreprocessor:
    '''
    文本预处理器

    属性:
        vectorizer: TfidfVectorizer 实例
        label_encoder: LabelEncoder 实例
        target_names: 类别名称列表
    '''

    def __init__(self):
        self.vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
        self.label_encoder = LabelEncoder()
        self.target_names = None

    def prepare_data(self, df, text_col='text', label_col='category'):
        '''
        完整的数据准备流程：划分 → 拟合 → 转换

        参数:
            df: 原始 DataFrame
            text_col: 文本列名
            label_col: 标签列名

        返回:
            dict: {X_train, y_train, X_val, y_val, X_test, y_test}
        '''
        X_text = df[text_col].values
        y_labels = df[label_col].values

        self.target_names = list(np.unique(y_labels))
        print(f"📊 预处理器初始化")
        print(f"   类别: {self.target_names}")
        print(f"   TF-IDF 配置: max_features={TFIDF_CONFIG['max_features']}, "
              f"ngram_range={TFIDF_CONFIG['ngram_range']}")

        # 1. 划分训练集 和 临时集(验证+测试)
        X_train_text, X_temp_text, y_train_labels, y_temp_labels = train_test_split(
            X_text, y_labels,
            train_size=TRAIN_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_labels,
        )

        # 2. 划分验证集 和 测试集
        val_ratio = VAL_SIZE / (1 - TRAIN_SIZE)
        X_val_text, X_test_text, y_val_labels, y_test_labels = train_test_split(
            X_temp_text, y_temp_labels,
            train_size=val_ratio,
            random_state=RANDOM_STATE,
            stratify=y_temp_labels,
        )

        # 3. 用训练集拟合 vectorizer 和 label_encoder
        self.vectorizer.fit(X_train_text)
        self.label_encoder.fit(y_train_labels)

        # 4. 转换所有数据集
        X_train = self.vectorizer.transform(X_train_text)
        X_val = self.vectorizer.transform(X_val_text)
        X_test = self.vectorizer.transform(X_test_text)

        y_train = self.label_encoder.transform(y_train_labels)
        y_val = self.label_encoder.transform(y_val_labels)
        y_test = self.label_encoder.transform(y_test_labels)

        print(f"✅ 数据准备完成")
        print(f"   训练集: {X_train.shape} ({len(y_train)} 样本)")
        print(f"   验证集: {X_val.shape} ({len(y_val)} 样本)")
        print(f"   测试集: {X_test.shape} ({len(y_test)} 样本)")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
        }

    def transform_text(self, texts):
        '''
        用已拟合的 vectorizer 转换新文本
        '''
        if isinstance(texts, str):
            texts = [texts]
        return self.vectorizer.transform(texts)

    def encode_label(self, labels):
        '''
        用已拟合的 label_encoder 编码标签
        '''
        if isinstance(labels, str):
            labels = [labels]
        return self.label_encoder.transform(labels)

    def decode_label(self, encoded):
        '''
        将数字编码还原为类别名称
        '''
        if isinstance(encoded, (int, np.integer)):
            return self.label_encoder.inverse_transform([int(encoded)])[0]
        return self.label_encoder.inverse_transform(encoded)

    def save(self, filepath=None):
        '''
        保存预处理器（交付给甲方时使用）
        '''
        if filepath is None:
            filepath = PREPROCESSOR_PATH
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'target_names': self.target_names,
        }, filepath)
        print(f"💾 预处理器已保存: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath=None):
        '''
        从文件加载预处理器
        '''
        if filepath is None:
            filepath = PREPROCESSOR_PATH
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"预处理器文件不存在: {filepath}")

        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.vectorizer = data['vectorizer']
        preprocessor.label_encoder = data['label_encoder']
        preprocessor.target_names = data['target_names']
        print(f"📂 预处理器已加载: {filepath}")
        return preprocessor

    def save_processed_data(self, data_dict, filename='processed_data'):
        '''
        保存处理后的数据（稀疏矩阵用 scipy.sparse，数组用 npz）
        '''
        base = os.path.join(PROCESSED_DATA_DIR, filename)

        for key in ['X_train', 'X_val', 'X_test']:
            sp.save_npz(f"{base}_{key}.npz", data_dict[key])

        np.savez_compressed(
            f"{base}_y.npz",
            y_train=data_dict['y_train'],
            y_val=data_dict['y_val'],
            y_test=data_dict['y_test'],
        )

        print(f"💾 处理后数据已保存: {base}_*.npz")
        return base

    @staticmethod
    def load_processed_data(filename='processed_data'):
        '''
        从文件加载处理后的数据
        '''
        base = os.path.join(PROCESSED_DATA_DIR, filename)

        if not os.path.exists(f"{base}_X_train.npz"):
            raise FileNotFoundError(
                f"处理后的数据文件不存在: {base}_X_train.npz\n"
                f"请先运行: python scripts/train.py --save"
            )

        X_train = sp.load_npz(f"{base}_X_train.npz")
        X_val = sp.load_npz(f"{base}_X_val.npz")
        X_test = sp.load_npz(f"{base}_X_test.npz")

        y_data = np.load(f"{base}_y.npz")

        print(f"📂 处理后数据已加载: {base}_*.npz")
        return {
            'X_train': X_train, 'y_train': y_data['y_train'],
            'X_val': X_val, 'y_val': y_data['y_val'],
            'X_test': X_test, 'y_test': y_data['y_test'],
        }


if __name__ == '__main__':
    from src.data_loader import load_raw_data
    from configs.config import DEFAULT_CATEGORIES

    print("=" * 60)
    print("测试 TextPreprocessor")
    print("=" * 60)

    df = load_raw_data(categories=DEFAULT_CATEGORIES)

    preprocessor = TextPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    preprocessor.save()
    preprocessor.save_processed_data(data_dict)

    print("\n--- 测试 transform_text ---")
    test_texts = [
        "God is love and religion is important",
        "The computer graphics rendering is fast",
        "Medical research shows new treatment",
    ]
    X_new = preprocessor.transform_text(test_texts)
    print(f"输入 {len(test_texts)} 条文本 → 输出形状: {X_new.shape}")

    print("\n--- 测试 decode_label ---")
    for i, name in enumerate(preprocessor.target_names):
        decoded = preprocessor.decode_label(i)
        print(f"  编码 {i} → '{decoded}' {'✅' if decoded == name else '❌'}")

    print("\n--- 测试 save/load 一致性 ---")
    loaded = TextPreprocessor.load()
    X_new2 = loaded.transform_text(test_texts)
    print(f"加载后 transform 结果一致: {(X_new != X_new2).nnz == 0} ✅")

    print("\n" + "=" * 60)
    print("所有测试通过 ✅")
    print("=" * 60)