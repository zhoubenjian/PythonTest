'''
测试模型工厂模块
'''
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import MODEL_CONFIGS
from src.model import get_model, save_model, load_model


class TestModelFactory(unittest.TestCase):

    def test_get_all_models(self):
        for model_type in MODEL_CONFIGS:
            model = get_model(model_type)
            self.assertIsNotNone(model)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            get_model('unknown_model_type')

    def test_save_and_load(self):
        import tempfile
        model = get_model('naive_bayes')

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name

        save_model(model, path)
        loaded = load_model(path)

        self.assertEqual(type(loaded).__name__, type(model).__name__)
        os.unlink(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)