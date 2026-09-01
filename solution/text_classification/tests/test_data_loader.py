'''
测试数据加载模块
'''
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import RAW_DATA_DIR, DEFAULT_CATEGORIES
from src.data_loader import load_raw_data, save_raw_data, load_from_csv


class TestDataLoader(unittest.TestCase):

    def test_load_raw_data(self):
        df = load_raw_data(categories=DEFAULT_CATEGORIES)
        self.assertGreater(len(df), 0)
        self.assertIn('text', df.columns)
        self.assertIn('category', df.columns)
        self.assertEqual(df['category'].nunique(), len(DEFAULT_CATEGORIES))

    def test_save_and_load_csv(self):
        df = load_raw_data(categories=DEFAULT_CATEGORIES)
        filepath = save_raw_data(df)
        self.assertTrue(os.path.exists(filepath))

        df2 = load_from_csv(filepath)
        self.assertEqual(len(df), len(df2))
        self.assertEqual(list(df.columns), list(df2.columns))

    def test_missing_csv(self):
        with self.assertRaises(FileNotFoundError):
            load_from_csv('nonexistent_file.csv')


if __name__ == '__main__':
    unittest.main(verbosity=2)