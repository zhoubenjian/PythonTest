'''
训练入口脚本
完整流程：加载数据 → 预处理 → 训练 → 保存交付物
'''
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import DEFAULT_CATEGORIES
from src.data_loader import load_raw_data, save_raw_data
from src.preprocessor import TextPreprocessor
from src.trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='文本分类模型训练')
    parser.add_argument(
        '--save', action='store_true',
        help='保存预处理器和所有模型到 models/ 目录'
    )
    parser.add_argument(
        '--save-raw', action='store_true',
        help='保存原始数据到 data/raw/'
    )
    parser.add_argument(
        '--categories', nargs='+', default=None,
        help='指定类别列表（默认使用配置中的 4 类）'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏠 文本分类 - 训练脚本")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据")
    categories = args.categories or DEFAULT_CATEGORIES
    df = load_raw_data(categories=categories)

    if args.save_raw:
        save_raw_data(df)

    # Step 2: 预处理
    print("\n[Step 2] 数据预处理")
    preprocessor = TextPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    # Step 3: 训练所有模型
    print("\n[Step 3] 模型训练")
    trainer = ModelTrainer()
    trainer.train_all(data_dict)

    # Step 4: 保存交付物
    if args.save:
        print("\n[Step 4] 保存交付物")
        preprocessor.save()
        preprocessor.save_processed_data(data_dict)
        trainer.save_all_models()

        print("\n" + "=" * 60)
        print("📦 交付物清单")
        print("=" * 60)
        print("models/")
        print("├── preprocessor.pkl          # 预处理器")
        for f in os.listdir(os.path.join(PROJECT_ROOT, 'models')):
            if f.endswith('_model.pkl'):
                print(f"├── {f}")
        print("=" * 60)

    print("\n✅ 训练完成！")
    print("   下一步: python scripts/evaluate.py")
    print("   或:     python scripts/predict.py")


if __name__ == '__main__':
    main()