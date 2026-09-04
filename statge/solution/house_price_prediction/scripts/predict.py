'''
预测入口脚本
Usage:
    python scripts/predict.py                             # 交互式预测
    python scripts/predict.py --model models/forest_model.pkl  # 使用指定模型
    python scripts/predict.py --batch                     # 批量预测演示
'''
import os
import sys
import argparse
import joblib
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.data_loader import load_raw_data
from src.preprocessor import DataPreprocessor
from src.predictor import HousePricePredictor
from configs.config import FEATURE_COLUMNS

def load_best_model(models_dir: str):
    """加载最佳模型（基于优先级选择）"""
    if not os.path.exists(models_dir):
        return None, None

    # 按优先级选择模型
    preferred_order = ["forest", "nn", "ridge", "linear"]
    for name in preferred_order:
        model_path = os.path.join(models_dir, f"{name}_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model, name

    # 如果没有偏好的，加载第一个
    for filename in os.listdir(models_dir):
        if filename.endswith("_model.pkl"):
            model = joblib.load(os.path.join(models_dir, filename))
            return model, filename.replace("_model.pkl", "")

    return None, None

def interactive_mode(predictor: HousePricePredictor):
    """交互式预测模式"""
    print("\n" + "=" * 60)
    print("🏠 房价预测 - 交互式模式")
    print("=" * 60)
    print("请输入房屋特征值（直接回车使用默认值 0）")
    print("输入 'quit' 或 'q' 退出\n")

    # 特征提示
    feature_hints = {
        "MedInc": "收入中位数 (例: 5.0)",
        "HouseAge": "房龄 (例: 30)",
        "AveRooms": "平均房间数 (例: 6)",
        "AveBedrms": "平均卧室数 (例: 3)",
        "Population": "人口 (例: 1000)",
        "AveOccup": "平均入住率 (例: 3)",
        "Latitude": "纬度 (例: 37.0)",
        "Longitude": "经度 (例: -122.0)",
    }

    while True:
        print("-" * 40)
        print("请输入特征值:")
        print("-" * 40)

        features = {}
        for col in FEATURE_COLUMNS:
            hint = feature_hints.get(col, "")
            try:
                value = input(f"  {col} ({hint}): ").strip()
                if value.lower() in ("quit", "q", "exit"):
                    print("👋 退出预测")
                    return
                features[col] = float(value) if value else 0.0
            except ValueError:
                print(f"⚠️ 无效输入，{col} 使用默认值 0")
                features[col] = 0.0

        # 预测
        try:
            price = predictor.predict_single(**features)
            print(f"\n💰 预测房价: ${price:,.2f} (单位: 万美元)")
        except Exception as e:
            print(f"❌ 预测出错: {e}")

def batch_demo(predictor: HousePricePredictor):
    """批量预测演示"""
    print("\n" + "=" * 60)
    print("📊 批量预测演示")
    print("=" * 60)

    # 构造示例数据
    demo_samples = [
        {
            "MedInc": 8.5, "HouseAge": 15, "AveRooms": 5,
            "AveBedrms": 2, "Population": 500, "AveOccup": 2.5,
            "Latitude": 37.7, "Longitude": -122.4,  # 旧金山
        },
        {
            "MedInc": 3.0, "HouseAge": 40, "AveRooms": 7,
            "AveBedrms": 4, "Population": 2000, "AveOccup": 4,
            "Latitude": 34.0, "Longitude": -118.2,  # 洛杉矶
        },
        {
            "MedInc": 1.5, "HouseAge": 50, "AveRooms": 4,
            "AveBedrms": 2, "Population": 800, "AveOccup": 3,
            "Latitude": 36.7, "Longitude": -119.4,  # 佛雷斯诺
        },
    ]

    # 批量预测
    predictions = predictor.predict_batch(demo_samples)

    for i, (sample, pred) in enumerate(zip(demo_samples, predictions)):
        print(f"\n🏠 样本 {i + 1}:")
        print(f"   收入中位数: {sample['MedInc']}")
        print(f"   房龄: {sample['HouseAge']} 年")
        print(f"   位置: ({sample['Latitude']}, {sample['Longitude']})")
        print(f"   💰 预测房价: ${pred:,.2f} 万美元")

def main():
    parser = argparse.ArgumentParser(description="房价预测")
    parser.add_argument(
        "--model", type=str, default=None,
        help="指定模型路径",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="运行批量预测演示",
    )

    args = parser.parse_args()

    # ============ 加载模型 ============
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return
        model = joblib.load(model_path)
        model_name = os.path.basename(model_path)
    else:
        models_dir = os.path.join(PROJECT_ROOT, "models")
        model, model_name = load_best_model(models_dir)
        if model is None:
            print("❌ 未找到训练好的模型")
            print("   请先运行: python scripts/train.py --save")
            return

    print(f"🧠 使用模型: {model_name}")

    # ============ 加载预处理器 ============
    # 使用新数据重新拟合预处理器
    df = load_raw_data()
    preprocessor = DataPreprocessor()
    preprocessor.prepare_data(df)

    # ============ 创建预测器 ============
    predictor = HousePricePredictor(model, preprocessor)

    # ============ 运行预测 ============
    if args.batch:
        batch_demo(predictor)
    else:
        interactive_mode(predictor)

if __name__ == "__main__":
    main()