'''
评估入口脚本
加载已保存的模型 → 对比评估 → 打印报告
'''
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import DEFAULT_CATEGORIES, MODEL_DIR
from src.preprocessor import TextPreprocessor
from src.trainer import ModelTrainer
from src.model import load_model
from src.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description='文本分类模型评估')
    parser.add_argument(
        '--model', type=str, default=None,
        help='指定模型类型（如 naive_bayes），默认评估所有'
    )
    parser.add_argument(
        '--categories', nargs='+', default=None,
        help='指定类别列表'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("📊 文本分类 - 评估脚本")
    print("=" * 60)

    # Step 1: 加载模型文件
    print("\n[Step 1] 加载模型")
    preprocessor = TextPreprocessor.load()

    model_files = [
        f for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')
    ]
    if not model_files:
        print("❌ 未找到模型文件，请先运行: python scripts/train.py --save")
        return

    trainer = ModelTrainer()
    for mf in model_files:
        model_type = mf.replace('_model.pkl', '')
        filepath = os.path.join(MODEL_DIR, mf)
        model = load_model(filepath)
        trainer.models[model_type] = model
        print(f"   已加载: {mf}")

    # Step 2: 加载已保存的处理数据
    print("\n[Step 2] 加载已处理数据")
    data_dict = TextPreprocessor.load_processed_data()
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    # Step 3: 评估
    print("\n[Step 3] 模型评估")
    evaluator = ModelEvaluator(target_names=preprocessor.target_names)

    if args.model:
        if args.model not in trainer.models:
            print(f"❌ 未找到模型 '{args.model}'")
            return
        model = trainer.models[args.model]
        evaluator.print_report(model, X_test, y_test)
        evaluator.print_confusion_matrix(model, X_test, y_test)
    else:
        best_type, best_model, _ = evaluator.compare_models(trainer.models, X_test, y_test)
        evaluator.print_report(best_model, X_test, y_test)
        evaluator.print_confusion_matrix(best_model, X_test, y_test)

    print("\n✅ 评估完成！")


if __name__ == '__main__':
    main()