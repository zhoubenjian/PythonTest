'''
测试训练器模块
'''
import os
import sys
import unittest
import tempfile
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_data
from src.preprocessor import TextPreprocessor
from src.trainer import ModelTrainer
from configs.config import DEFAULT_CATEGORIES, MODEL_CONFIGS


class TestModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = load_raw_data(categories=DEFAULT_CATEGORIES)
        cls.preprocessor = TextPreprocessor()
        cls.data_dict = cls.preprocessor.prepare_data(df)
        cls.trainer = ModelTrainer()
        cls.trainer.train_all(cls.data_dict)

    def test_all_models_trained(self):
        for model_type in MODEL_CONFIGS:
            self.assertIn(model_type, self.trainer.models)
            self.assertIn(model_type, self.trainer.scores)

    def test_metrics_exist(self):
        for model_type, score in self.trainer.scores.items():
            self.assertIn('train_acc', score)
            self.assertIn('val_acc', score)
            self.assertIn('val_f1', score)

    def test_train_accuracy_range(self):
        for model_type, score in self.trainer.scores.items():
            self.assertGreaterEqual(score['train_acc'], 0.0)
            self.assertLessEqual(score['train_acc'], 1.0)
            self.assertGreaterEqual(score['val_f1'], 0.0)
            self.assertLessEqual(score['val_f1'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)