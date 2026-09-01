'''
模型训练器
负责训练单个/多个模型，计算指标，保存模型
'''
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import MODEL_DIR, MODEL_CONFIGS, AVERAGE_METHOD
from src.model import get_model, save_model


class ModelTrainer:
    '''
    模型训练器

    属性:
        models: dict，存储已训练的模型 {model_type: model}
        scores: dict，存储各模型指标 {model_type: {train_acc, val_acc, ...}}
    '''

    def __init__(self):
        self.models = {}
        self.scores = {}

    def train(self, model_type, data_dict):
        '''
        训练单个模型

        参数:
            model_type: 模型 key (如 'naive_bayes')
            data_dict: 预处理后的数据字典

        返回:
            model: 训练好的模型
        '''
        model = get_model(model_type)

        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']

        print(f"\n🏋️  训练 '{model_type}' ...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average=AVERAGE_METHOD)

        self.models[model_type] = model
        self.scores[model_type] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_f1': val_f1,
        }

        print(f"   训练集准确率: {train_acc:.4f}")
        print(f"   验证集准确率: {val_acc:.4f}")
        print(f"   验证集 F1({AVERAGE_METHOD}): {val_f1:.4f}")

        return model

    def train_all(self, data_dict):
        '''
        训练配置中的所有模型，并打印对比表
        '''
        print("\n" + "=" * 60)
        print("开始训练所有模型")
        print("=" * 60)

        for model_type in MODEL_CONFIGS:
            self.train(model_type, data_dict)

        self._print_comparison()

    def _print_comparison(self):
        '''打印模型性能对比表'''
        print("\n" + "=" * 60)
        print("📊 模型性能对比 (验证集)")
        print("=" * 60)
        print(f"{'模型':<15} {'训练准确率':>12} {'验证准确率':>12} {'验证F1':>12}")
        print("-" * 54)
        for model_type, score in self.scores.items():
            print(
                f"{model_type:<15} "
                f"{score['train_acc']:>12.4f} "
                f"{score['val_acc']:>12.4f} "
                f"{score['val_f1']:>12.4f}"
            )
        print("=" * 60)

        best_type = max(self.scores, key=lambda k: self.scores[k]['val_f1'])
        best_f1 = self.scores[best_type]['val_f1']
        print(f"🏆 最佳模型: '{best_type}' (F1={best_f1:.4f})")

    def save_all_models(self):
        '''
        保存所有已训练的模型到 models/ 目录
        '''
        saved_paths = {}
        for model_type, model in self.models.items():
            filepath = os.path.join(MODEL_DIR, f"{model_type}_model.pkl")
            save_model(model, filepath)
            saved_paths[model_type] = filepath
        return saved_paths

    def save_best_model(self):
        '''
        只保存验证集 F1 最高的模型
        '''
        if not self.scores:
            raise ValueError("没有已训练的模型，请先调用 train() 或 train_all()")

        best_type = max(self.scores, key=lambda k: self.scores[k]['val_f1'])
        filepath = os.path.join(MODEL_DIR, f"{best_type}_model.pkl")
        save_model(self.models[best_type], filepath)
        print(f"🏆 最佳模型已保存: {filepath}")
        return filepath, best_type

if __name__ == '__main__':
    from src.data_loader import load_raw_data
    from src.preprocessor import TextPreprocessor
    from configs.config import DEFAULT_CATEGORIES

    print("=" * 60)
    print("测试 ModelTrainer")
    print("=" * 60)

    df = load_raw_data(categories=DEFAULT_CATEGORIES)
    preprocessor = TextPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    trainer = ModelTrainer()
    trainer.train_all(data_dict)

    # 保存预处理器和模型文件
    preprocessor.save()
    preprocessor.save_processed_data(data_dict)
    trainer.save_all_models()

    print("\n" + "=" * 60)
    print("训练完成 ✅")
    print("=" * 60)