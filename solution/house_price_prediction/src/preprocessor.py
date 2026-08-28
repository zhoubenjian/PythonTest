'''
数据预处理模块
对原始数据划分，特征工程，标准化等
'''
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径（必须在导入 configs 之前）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    FEATURE_COLUMNS, TARGET_COLUMN,
    TRAIN_SIZE, VAL_SIZE, RANDOM_STATE,
    PROCESSED_DATA_DIR,
)


'''
数据预处理类
'''
class DataPreprocessor:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN


    def prepare_data(self, df: pd.DataFrame) -> dict:
        """
        完整的数据预处理流程
        :param df: 原始数据 DataFrame
        :return: 包含训练集、验证集、测试集的字典
        """
        print("🔧 开始数据预处理...")

        # 1.特征，标签分离
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        print(f'特征形状: {X.shape}')
        print(f'标签形状: {y.shape}')

        # 2.数据划分训练集，验证集，测试集
        # 先划分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            train_size=TRAIN_SIZE + VAL_SIZE,
            test_size=1 - (TRAIN_SIZE + VAL_SIZE),
            random_state=RANDOM_STATE
        )

        # 再从剩余中划分出验证集
        relative_val_size = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val_size,
            random_state=RANDOM_STATE
        )

        print(f'划分结果：训练={len(X_train)}，验证={len(X_val)}，测试={len(X_test)}')

        # 3.标准化特征
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)

        # 标准化标签
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

        print('✅ 数据标准化完成')

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            # 保留原始数据用于反标准化
            "X_train_raw": X_train,
            "X_val_raw": X_val,
            "X_test_raw": X_test
        }


    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        将标准化的标签还原为原始尺度
        :param y_pred_scaled: 标准化后的标签
        :return: 原始尺度的标签
        """
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled).ravel()


    def save_processed_data(self, data_dict: dict) -> None:
        """
        保存预处理后的数据
        :param data_dict: 返回的数据字典
        :return:
        """
        filepath = os.path.join(PROCESSED_DATA_DIR, "processed_data.npz")
        np.savez_compressed(
            filepath,
            X_train=data_dict["X_train"],
            y_train=data_dict["y_train"],
            X_val=data_dict["X_val"],
            y_val=data_dict["y_val"],
            X_test=data_dict["X_test"],
            y_test=data_dict["y_test"],
        )
        print(f"💾 处理后的数据已保存: {filepath}")


    @staticmethod
    def load_preprocessed_data(filepath: str) -> dict:
        """
        加载预处理后的数据
        :param filepath: 数据文件路径
        :return: 返回的数据字典
        """
        print(f"📂 加载处理后的数据: {filepath}")
        data = np.load(filepath)
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
        }


if __name__ == "__main__":
    # 测试预处理流程
    from data_loader import load_raw_data

    # 加载数据
    df = load_raw_data()

    # 预处理
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # 保存处理后的数据
    preprocessor.save_processed_data(data_dict)

    # 验证数据形状
    print("\n📊 数据形状验证:")
    for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        print(f"{key}: {data_dict[key].shape}")

    print("\n✅ 数据预处理测试通过！")


