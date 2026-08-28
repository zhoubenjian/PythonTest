'''
训练模块
负责训练模型和调优
'''
import os
import sys
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目根目录到路径（必须在导入 configs 之前）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import MODEL_CONFIGS


class ModelTrainer:
    """模型训练器"""

    def __init__(self):
        self.models = {}          # 存储训练好的模型
        self.training_history = {}  # 存储训练历史

    def train(self, model_type: str, data_dict: dict) -> dict:
        """
        训练单个模型
        :param model_type: 模型类型
        :param data_dict: 数据字典
        :return: 训练结果
        """
        from model import get_model

        print(f"\n{'='*50}")
        print(f"🚀 开始训练: {model_type}")
        print(f"{'='*50}")

        # 获取数据
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"]

        # 创建并训练模型
        model = get_model(model_type)
        model.fit(X_train, y_train)

        # 评估
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_metrics = self._calculate_metrics(y_train, train_pred)
        val_metrics = self._calculate_metrics(y_val, val_pred)

        print(f"\n📊 训练集指标:")
        for metric, value in train_metrics.items():
            print(f"   {metric}: {value:.4f}")

        print(f"\n📊 验证集指标:")
        for metric, value in val_metrics.items():
            print(f"   {metric}: {value:.4f}")

        # 保存模型
        self.models[model_type] = model
        self.training_history[model_type] = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        return {
            "model_type": model_type,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def train_all(self, data_dict: dict) -> list:
        """
        训练所有模型
        Args:
            data_dict: 数据字典
        Returns:
            list: 所有模型的训练结果
        """
        results = []
        for model_type in MODEL_CONFIGS:
            result = self.train(model_type, data_dict)
            results.append(result)

        # 打印汇总对比
        print(f"\n{'='*50}")
        print("📈 模型性能对比（验证集）")
        print(f"{'='*50}")
        print(f"{'模型':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 56)
        for model_type, history in self.training_history.items():
            val = history["val_metrics"]
            rmse = np.sqrt(val["mse"])
            print(f"{model_type:<20} {rmse:<12.4f} {val['mae']:<12.4f} {val['r2']:<12.4f}")

        return results

    def save_model(self, model_type: str, save_path: str) -> None:
        """
        保存训练好的模型
        Args:
            model_type: 模型类型
            save_path: 保存路径
        """
        if model_type not in self.models:
            raise ValueError(f"未找到已训练的模型: {model_type}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.models[model_type], save_path)
        print(f"💾 模型已保存: {save_path}")

    def save_all_models(self, save_dir: str) -> None:
        """
        保存所有训练好的模型
        Args:
            save_dir: 保存目录
        """
        for model_type in self.models:
            save_path = os.path.join(save_dir, f"{model_type}_model.pkl")
            self.save_model(model_type, save_path)

    @staticmethod
    def load_model(model_path: str):
        """
        加载训练好的模型
        Args:
            model_path: 模型文件路径
        Returns:
            模型实例
        """
        return joblib.load(model_path)

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        计算评估指标
        Args:
            y_true: 真实值
            y_pred: 预测值
        Returns:
            dict: 指标字典
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
        }

if __name__ == "__main__":
    # 测试训练流程
    from data_loader import load_raw_data
    from preprocessor import DataPreprocessor

    # 1. 加载数据
    print("📊 加载数据...")
    df = load_raw_data()

    # 2. 数据预处理
    print("\n🔧 数据预处理...")
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # 3. 训练模型
    print("\n🚀 开始训练...")
    trainer = ModelTrainer()
    results = trainer.train("ridge", data_dict)

    # 4. 保存模型
    print("\n💾 保存模型...")
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    save_path = os.path.join(save_dir, "ridge_model.pkl")
    trainer.save_model("ridge", save_path)

    print("\n✅ 训练器测试通过！")