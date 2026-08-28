'''
评估入口脚本
Usage:
    python scripts/evaluate.py                           # 评估所有模型
    python scripts/evaluate.py --model-path models/ridge_model.pkl  # 评估指定模型
'''
import os
import sys
import argparse
import joblib

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.data_loader import load_raw_data
from src.preprocessor import DataPreprocessor
from src.evaluator import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description="房价预测 - 模型评估")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="指定模型路径（不指定则评估所有已保存的模型）",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="指定处理后的数据路径",
    )

    args = parser.parse_args()

    # ============ Step 1: 准备数据 ============
    if args.data_path and os.path.exists(args.data_path):
        # 加载已处理的数据
        data_dict = DataPreprocessor.load_processed_data(args.data_path)
    else:
        # 重新加载和处理数据
        df = load_raw_data()
        preprocessor = DataPreprocessor()
        data_dict = preprocessor.prepare_data(df)

    # ============ Step 2: 加载模型并评估 ============
    evaluator = ModelEvaluator()

    if args.model_path:
        # 评估单个模型
        print(f"\n📊 评估模型: {args.model_path}")
        model = joblib.load(args.model_path)
        result = evaluator.evaluate(
            model, data_dict["X_test"], data_dict["y_test"]
        )
        print(f"   MSE:  {result['mse']:.4f}")
        print(f"   RMSE: {result['rmse']:.4f}")
        print(f"   MAE:  {result['mae']:.4f}")
        print(f"   R²:   {result['r2']:.4f}")
    else:
        # 评估所有模型
        models_dir = os.path.join(PROJECT_ROOT, "models")
        if not os.path.exists(models_dir):
            print(f"❌ 未找到模型目录: {models_dir}")
            print("   请先运行训练脚本并保存模型: python scripts/train.py --save")
            return

        # 加载所有模型
        models_dict = {}
        for filename in os.listdir(models_dir):
            if filename.endswith("_model.pkl"):
                model_name = filename.replace("_model.pkl", "")
                model_path = os.path.join(models_dir, filename)
                models_dict[model_name] = joblib.load(model_path)

        if not models_dict:
            print("❌ 未找到任何已保存的模型")
            return

        # 对比评估
        evaluator.compare_models(
            models_dict, data_dict["X_test"], data_dict["y_test"]
        )

    print("\n✅ 评估完成！")

if __name__ == "__main__":
    main()