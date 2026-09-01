'''
测试预测器模块
'''
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predictor import TextClassifierPredictor
from configs.config import PREPROCESSOR_PATH, MODEL_DIR


class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(PREPROCESSOR_PATH):
            raise unittest.SkipTest(
                "模型文件不存在，请先运行 train.py --save"
            )
        cls.predictor = TextClassifierPredictor()

    def test_predict_single(self):
        result = self.predictor.predict("God and religion are important")
        self.assertIn('category', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_predict_batch(self):
        texts = [
            "God is love",
            "Computer graphics rendering",
            "Medical treatment research",
            "Baseball game score",
        ]
        results = self.predictor.predict_batch(texts)
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIn('category', r)
            self.assertIn('confidence', r)

    def test_probabilities_sum_to_one(self):
        result = self.predictor.predict("Test text for prediction")
        total = sum(result['probabilities'].values())
        self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)