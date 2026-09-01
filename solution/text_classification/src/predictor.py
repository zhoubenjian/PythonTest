'''
预测器
支持单条/批量文本预测，交互式预测，适配甲方交付
'''
import os
import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import MODEL_DIR, PREPROCESSOR_PATH
from src.model import load_model
from src.preprocessor import TextPreprocessor


class TextClassifierPredictor:
    '''
    文本分类预测器

    使用方式:
        predictor = TextClassifierPredictor(model_path, preprocessor_path)
        result = predictor.predict("This is a computer graphics issue")
    '''

    def __init__(self, model_path=None, preprocessor_path=None):
        if preprocessor_path is None:
            preprocessor_path = PREPROCESSOR_PATH
        self.preprocessor = TextPreprocessor.load(preprocessor_path)

        if model_path is None:
            model_path = self._find_best_model()
        self.model = load_model(model_path)

    def _find_best_model(self):
        '''在 models/ 目录下找到最佳模型（F1 最高的）'''
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"模型目录不存在: {MODEL_DIR}")

        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')]
        if not model_files:
            raise FileNotFoundError(
                f"未找到模型文件，请先运行 train.py --save\n"
                f"目录: {MODEL_DIR}"
            )

        scoring_keys = ['svm', 'logistic', 'naive_bayes', 'forest']
        for key in scoring_keys:
            for f in model_files:
                if f.startswith(key):
                    path = os.path.join(MODEL_DIR, f)
                    print(f"自动选择模型: {path}")
                    return path

        path = os.path.join(MODEL_DIR, model_files[0])
        print(f"自动选择模型: {path}")
        return path

    def predict(self, text):
        '''
        单条文本预测

        参数:
            text: str，输入文本

        返回:
            dict: {category, confidence, probabilities}
        '''
        X = self.preprocessor.transform_text([text])
        probas = self.model.predict_proba(X)[0]

        pred_idx = np.argmax(probas)
        category = self.preprocessor.decode_label(pred_idx)
        confidence = float(probas[pred_idx])

        proba_dict = {
            self.preprocessor.decode_label(i): float(p)
            for i, p in enumerate(probas)
        }

        return {
            'category': category,
            'confidence': confidence,
            'probabilities': proba_dict,
        }

    def predict_batch(self, texts):
        '''
        批量文本预测

        参数:
            texts: List[str]

        返回:
            List[dict]
        '''
        X = self.preprocessor.transform_text(texts)
        probas = self.model.predict_proba(X)
        predictions = []

        for i in range(len(texts)):
            pred_idx = np.argmax(probas[i])
            category = self.preprocessor.decode_label(pred_idx)
            confidence = float(probas[i][pred_idx])

            proba_dict = {
                self.preprocessor.decode_label(j): float(probas[i][j])
                for j in range(len(probas[i]))
            }

            predictions.append({
                'text': texts[i][:100] + ('...' if len(texts[i]) > 100 else ''),
                'category': category,
                'confidence': confidence,
                'probabilities': proba_dict,
            })

        return predictions

    def interactive(self):
        '''
        交互式预测模式
        '''
        print("\n" + "=" * 60)
        print("🎯 交互式文本分类预测")
        print("   输入文本后回车预测，输入 q 退出")
        print(f"   类别: {self.preprocessor.target_names}")
        print("=" * 60)

        while True:
            try:
                text = input("\n请输入文本> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 退出")
                break

            if text.lower() in ('q', 'quit', 'exit'):
                print("👋 退出")
                break

            if not text:
                continue

            result = self.predict(text)
            print(f"预测类别: {result['category']}")
            print(f"置信度:   {result['confidence']:.2%}")
            print(f"各类别概率:")
            for cat, prob in sorted(result['probabilities'].items(),
                                    key=lambda x: -x[1]):
                bar = '█' * int(prob * 20)
                print(f"  {cat:<30} {bar} {prob:.2%}")


if __name__ == '__main__':
    print("=" * 60)
    print("测试 TextClassifierPredictor")
    print("=" * 60)

    if not os.path.exists(PREPROCESSOR_PATH):
        print("⚠️  预处理器不存在，请先运行 train.py --save")
        print("   跳过 predictor 测试")
    else:
        predictor = TextClassifierPredictor()

        test_texts = [
            "God and religion are very important in our lives",
            "Computer graphics and rendering with GPU acceleration",
            "Medical study shows new treatment for cancer patients",
            "Baseball game score and player statistics",
        ]

        print("\n--- 批量预测 ---")
        results = predictor.predict_batch(test_texts)
        for r in results:
            print(f"\n文本: {r['text'][:60]}...")
            print(f"  → {r['category']} (置信度: {r['confidence']:.2%})")

        print("\n" + "=" * 60)
        print("预测器测试完成 ✅")
        print("=" * 60)