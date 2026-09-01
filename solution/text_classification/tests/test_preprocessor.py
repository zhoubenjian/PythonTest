'''
测试预处理器模块
'''
import os
import sys
import unittest
import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_data
from src.preprocessor import TextPreprocessor
from configs.config import DEFAULT_CATEGORIES


class TestTextPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_raw_data(categories=DEFAULT_CATEGORIES)
        cls.preprocessor = TextPreprocessor()
        cls.data_dict = cls.preprocessor.prepare_data(cls.df)

    def test_prepare_data_shapes(self):
        self.assertIn('X_train', self.data_dict)
        self.assertIn('X_val', self.data_dict)
        self.assertIn('X_test', self.data_dict)

        X_train = self.data_dict['X_train']
        self.assertTrue(sp.issparse(X_train))
        self.assertEqual(X_train.shape[1], self.preprocessor.vectorizer.max_features)

    def test_data_split_ratios(self):
        total = len(self.df)
        train_ratio = len(self.data_dict['y_train']) / total
        val_ratio = len(self.data_dict['y_val']) / total
        test_ratio = len(self.data_dict['y_test']) / total

        self.assertAlmostEqual(train_ratio, 0.8, delta=0.02)
        self.assertAlmostEqual(val_ratio, 0.1, delta=0.02)
        self.assertAlmostEqual(test_ratio, 0.1, delta=0.02)

    def test_transform_text(self):
        texts = ["Hello world", "Computer graphics"]
        X = self.preprocessor.transform_text(texts)
        self.assertTrue(sp.issparse(X))
        self.assertEqual(X.shape[0], 2)

    def test_transform_single_text(self):
        X = self.preprocessor.transform_text("Hello world")
        self.assertEqual(X.shape[0], 1)

    def test_decode_label(self):
        for i, name in enumerate(self.preprocessor.target_names):
            decoded = self.preprocessor.decode_label(i)
            self.assertEqual(decoded, name)

    def test_save_and_load(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        self.preprocessor.save(path)
        loaded = TextPreprocessor.load(path)

        self.assertEqual(loaded.target_names, self.preprocessor.target_names)

        X_new = loaded.transform_text(["test graphics"])
        X_old = self.preprocessor.transform_text(["test graphics"])
        self.assertEqual((X_new != X_old).nnz, 0)

        os.unlink(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)