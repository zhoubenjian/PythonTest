'''
测试评估器模块
'''
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_data
from src.preprocessor import TextPreprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from configs.config import DEFAULT_CATEGORIES


class TestModelEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = load_raw_data(categories=DEFAULT_CATEGORIES)
        cls.preprocessor = TextPreprocessor()
        cls.data_dict = cls.preprocessor.prepare_data(df)

        cls.trainer = ModelTrainer()
        cls.trainer.train_all(cls.data_dict)

        cls.evaluator = ModelEvaluator(target_names=cls.preprocessor.target_names)

    def test_evaluate_single_model(self):
        model = self.trainer.models['naive_bayes']
        metrics = self.evaluator.evaluate(
            model,
            self.data_dict['X_test'],
            self.data_dict['y_test'],
        )
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)

    def test_compare_models(self):
        best_type, best_model, all_metrics = self.evaluator.compare_models(
            self.trainer.models,
            self.data_dict['X_test'],
            self.data_dict['y_test'],
        )
        self.assertIn(best_type, self.trainer.models)
        self.assertEqual(len(all_metrics), len(self.trainer.models))


if __name__ == '__main__':
    unittest.main(verbosity=2)