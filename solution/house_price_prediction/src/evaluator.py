'''
评估器模块
负责模型评估和结果分析
'''
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class ModelEvaluator:
    """模型评估器"""

    @staticmethod
    def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
        """
        评估单个模型
        Args:
            model: 已训练的模型
            X: 特征数据
            y: 真实标签
        Returns:
            dict: 评估指标
        """
        y_pred = model.predict(X)

        # MSE(均方误差，越小越好)
        mse = mean_squared_error(y, y_pred)
        # MAE(均绝对误差，越小越好)
        mae = mean_absolute_error(y, y_pred)
        # R²(模型解释方差的比例，越接近1越好)
        r2 = r2_score(y, y_pred)

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,
            "r2": r2,
            "predictions": y_pred,
        }

    @staticmethod
    def compare_models(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        对比多个模型的性能
        Args:
            models_dict: {model_name: model} 字典
            X_test: 测试特征
            y_test: 测试标签
        Returns:
            dict: 各模型的评估结果
        """
        results = {}

        print(f"\n{'='*60}")
        print("📊 测试集模型性能对比")
        print(f"{'='*60}")
        print(f"{'模型':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 68)

        for model_name, model in models_dict.items():
            result = ModelEvaluator.evaluate(model, X_test, y_test)
            results[model_name] = result

            print(
                f"{model_name:<20} "
                f"{result['mse']:<12.4f} "
                f"{result['rmse']:<12.4f} "
                f"{result['mae']:<12.4f} "
                f"{result['r2']:<12.4f}"
            )

        # 找出最佳模型
        best_model = max(results.keys(), key=lambda k: results[k]["r2"])
        print(f"\n🏆 最佳模型 (R² 最高): {best_model} (R²={results[best_model]['r2']:.4f})")

        return results

    @staticmethod
    def get_error_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        误差分析
        Args:
            y_true: 真实值
            y_pred: 预测值
        Returns:
            dict: 误差分析结果
        """
        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        return {
            "mean_error": np.mean(errors),
            "median_error": np.median(errors),
            "max_error": np.max(abs_errors),
            "min_error": np.min(abs_errors),
            "mean_abs_error": np.mean(abs_errors),
            "error_std": np.std(errors),
            "over_predictions": int(np.sum(errors < 0)),    # 预测偏高
            "under_predictions": int(np.sum(errors > 0)),   # 预测偏低
        }


if __name__ == "__main__":
    # 测试评估流程
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
    trainer.train("ridge", data_dict)
    trainer.train("forest", data_dict)

    # 3. 评估模型
    print("\n📊 评估模型...")
    models_dict = {
        "ridge": trainer.models["ridge"],
        "forest": trainer.models["forest"],
    }

    evaluator = ModelEvaluator()
    results = evaluator.compare_models(
        models_dict, data_dict["X_test"], data_dict["y_test"]
    )

    # 4. 误差分析
    print("\n🔍 误差分析 (RandomForest):")
    forest_result = results["forest"]
    error_analysis = evaluator.get_error_analysis(
        data_dict["y_test"], forest_result["predictions"]
    )

    for key, value in error_analysis.items():
        print(f"   {key}: {value:.4f}")

    print("\n✅ 评估器测试通过！")