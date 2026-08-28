'''
预测器模块
负责对新数据进行预测
'''
import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class HousePricePredictor:
    """房价预测器"""

    def __init__(self, model, preprocessor):
        """
        初始化预测器
        Args:
            model: 已训练的模型
            preprocessor: 数据预处理器（用于反标准化）
        """
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, features) -> np.ndarray:
        """
        预测房价
        Args:
            features: 输入特征，支持多种格式
                - dict: 单个样本特征字典
                - pd.DataFrame: 多个样本
                - np.ndarray: 已处理的特征数组
        Returns:
            np.ndarray: 预测的房价（原始尺度）
        """
        # 1. 转换输入格式
        if isinstance(features, dict):
            # 单个样本字典 → DataFrame
            df = pd.DataFrame([features])
            X = self.preprocessor.scaler_X.transform(df)
        elif isinstance(features, pd.DataFrame):
            X = self.preprocessor.scaler_X.transform(features)
        elif isinstance(features, np.ndarray):
            X = features
        else:
            raise ValueError(f"不支持的输入格式: {type(features)}")

        # 2. 预测
        y_pred_scaled = self.model.predict(X)

        # 3. 反标准化回原始尺度
        y_pred = self.preprocessor.inverse_transform_y(y_pred_scaled)

        return y_pred

    def predict_single(self, **kwargs) -> float:
        """
        预测单个样本的房价
        Args:
            **kwargs: 特征名=值，如 MedInc=5.0, HouseAge=30
        Returns:
            float: 预测房价
        """
        # 验证特征名
        valid_features = self.preprocessor.feature_columns
        for key in kwargs:
            if key not in valid_features:
                raise ValueError(
                    f"无效的特征名: {key}\n"
                    f"可用特征: {valid_features}"
                )

        features = {col: kwargs.get(col, 0) for col in valid_features}
        prediction = self.predict(features)
        return float(prediction[0])

    def predict_batch(self, feature_list: list) -> np.ndarray:
        """
        批量预测
        Args:
            feature_list: 特征字典列表
        Returns:
            np.ndarray: 预测结果数组
        """
        df = pd.DataFrame(feature_list)
        return self.predict(df)


if __name__ == "__main__":
    # 测试预测流程
    from data_loader import load_raw_data
    from preprocessor import DataPreprocessor
    from trainer import ModelTrainer

    # 1. 加载和预处理数据
    df = load_raw_data()
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # 2. 训练模型
    print("\n🚀 训练模型...")
    trainer = ModelTrainer()
    trainer.train("forest", data_dict)

    # 3. 创建预测器
    predictor = HousePricePredictor(trainer.models["forest"], preprocessor)

    # 4. 单样本预测
    print("\n📊 单样本预测测试:")
    sample = {
        "MedInc": 8.5,
        "HouseAge": 15,
        "AveRooms": 5,
        "AveBedrms": 2,
        "Population": 500,
        "AveOccup": 2.5,
        "Latitude": 37.7,
        "Longitude": -122.4,
    }
    price = predictor.predict_single(**sample)
    print(f"   旧金山示例房价预测: ${price:,.2f} 万美元")

    # 5. 批量预测
    print("\n📊 批量预测测试:")
    batch_samples = [
        {
            "MedInc": 8.5, "HouseAge": 15, "AveRooms": 5,
            "AveBedrms": 2, "Population": 500, "AveOccup": 2.5,
            "Latitude": 37.7, "Longitude": -122.4,
        },
        {
            "MedInc": 3.0, "HouseAge": 40, "AveRooms": 7,
            "AveBedrms": 4, "Population": 2000, "AveOccup": 4,
            "Latitude": 34.0, "Longitude": -118.2,
        },
    ]
    predictions = predictor.predict_batch(batch_samples)
    for i, pred in enumerate(predictions):
        print(f"   样本 {i+1}: ${pred:,.2f} 万美元")

    print("\n✅ 预测器测试通过！")