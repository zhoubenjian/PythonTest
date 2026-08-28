'''
训练入口脚本
Usage:
    python scripts/train.py                     # 训练所有模型
    python scripts/train.py --model ridge       # 训练指定模型
    python scripts/train.py --model nn --save   # 训练并保存模型
'''
import os
import sys
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# 关键：添加 src 目录到路径，使 trainer.py 内部的 from model import 能找到模块
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from src.data_loader import load_raw_data
from src.preprocessor import DataPreprocessor
from src.trainer import ModelTrainer
from configs.config import MODEL_CONFIGS

def main():
    parser = argparse.ArgumentParser(description="房价预测 - 模型训练")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="指定训练的模型类型（不指定则训练所有）",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="是否保存训练好的模型",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default=os.path.join(PROJECT_ROOT, "models"),
        help="模型保存目录",
    )

    args = parser.parse_args()

    # ============ Step 1: 加载数据 ============
    print("=" * 60)
    print("📊 Step 1: 加载数据")
    print("=" * 60)
    df = load_raw_data()

    # ============ Step 2: 数据预处理 ============
    print("\n" + "=" * 60)
    print("🔧 Step 2: 数据预处理")
    print("=" * 60)
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # 保存处理后的数据
    preprocessor.save_processed_data(data_dict)

    # ============ Step 3: 模型训练 ============
    print("\n" + "=" * 60)
    print("🚀 Step 3: 模型训练")
    print("=" * 60)
    trainer = ModelTrainer()

    if args.model:
        # 训练指定模型
        results = trainer.train(args.model, data_dict)
    else:
        # 训练所有模型
        results = trainer.train_all(data_dict)

    # ============ Step 4: 保存模型 ============
    if args.save:
        print("\n" + "=" * 60)
        print("💾 Step 4: 保存模型")
        print("=" * 60)
        os.makedirs(args.save_dir, exist_ok=True)

        if args.model:
            save_path = os.path.join(args.save_dir, f"{args.model}_model.pkl")
            trainer.save_model(args.model, save_path)
        else:
            trainer.save_all_models(args.save_dir)

    print("\n✅ 训练完成！")
    return results


if __name__ == "__main__":
    main()