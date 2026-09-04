'''
预测入口脚本
加载已保存的模型 → 交互式/单条/批量预测
'''
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predictor import TextClassifierPredictor

def main():
    parser = argparse.ArgumentParser(description='文本分类预测')
    parser.add_argument(
        '--text', type=str, default=None,
        help='单条文本预测'
    )
    parser.add_argument(
        '--batch', action='store_true',
        help='批量 demo 预测（4 条示例文本）'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='指定模型类型（如 naive_bayes），默认自动选择'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🎯 文本分类 - 预测脚本")
    print("=" * 60)

    predictor = TextClassifierPredictor()

    if args.model:
        model_path = os.path.join(PROJECT_ROOT, 'models', f"{args.model}_model.pkl")
        if not os.path.exists(model_path):
            print(f"❌ 未找到模型文件: {model_path}")
            return
        predictor.model = __import__('joblib').load(model_path)
        print(f"   已切换模型: {args.model}")

    if args.text:
        print(f"\n📝 单条预测")
        print(f"   输入: {args.text}")
        result = predictor.predict(args.text)
        print(f"\n   预测类别: {result['category']}")
        print(f"   置信度:   {result['confidence']:.2%}")
        print(f"   各类别概率:")
        for cat, prob in sorted(result['probabilities'].items(),
                                key=lambda x: -x[1]):
            bar = '█' * int(prob * 20)
            print(f"     {cat:<30} {bar} {prob:.2%}")

    elif args.batch:
        test_texts = [
            "God and religion are very important in our lives",
            "Computer graphics and rendering with GPU acceleration",
            "Medical study shows new treatment for cancer patients",
            "Baseball game score and player statistics",
        ]
        print(f"\n📦 批量预测 ({len(test_texts)} 条)")
        results = predictor.predict_batch(test_texts)
        for r in results:
            print(f"\n文本: {r['text'][:60]}")
            print(f"  → {r['category']} (置信度: {r['confidence']:.2%})")

    else:
        predictor.interactive()


if __name__ == '__main__':
    main()