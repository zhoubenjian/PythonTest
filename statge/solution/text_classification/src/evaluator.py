'''
模型评估器
负责单模型评估、多模型对比、分类报告、混淆矩阵
'''
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import AVERAGE_METHOD, MODEL_CONFIGS


class ModelEvaluator:
    '''
    模型评估器
    '''

    def __init__(self, target_names=None):
        self.target_names = target_names

    def evaluate(self, model, X, y):
        '''
        单模型评估

        返回:
            dict: {accuracy, f1, precision, recall, y_pred}
        '''
        y_pred = model.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred, average=AVERAGE_METHOD),
            'precision': precision_score(y, y_pred, average=AVERAGE_METHOD, zero_division=0),
            'recall': recall_score(y, y_pred, average=AVERAGE_METHOD, zero_division=0),
            'y_pred': y_pred,
        }
        return metrics

    def print_report(self, model, X, y):
        '''
        打印详细分类报告
        '''
        y_pred = model.predict(X)

        print(f"\n📊 分类报告:")
        print(f"  准确率: {accuracy_score(y, y_pred):.4f}")
        print(f"  F1({AVERAGE_METHOD}): {f1_score(y, y_pred, average=AVERAGE_METHOD):.4f}")
        print()
        print(classification_report(
            y, y_pred,
            target_names=self.target_names,
            digits=4,
            zero_division=0,
        ))

    def print_confusion_matrix(self, model, X, y):
        '''
        打印混淆矩阵
        '''
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        print(f"\n🔢 混淆矩阵 (行=真实, 列=预测):")
        header = "        " + " ".join(f"{t:>12}" for t in self.target_names)
        print(header)
        for i, row in enumerate(cm):
            label = self.target_names[i] if self.target_names else str(i)
            print(f"  {label:<8}" + " ".join(f"{v:>12}" for v in row))

    def compare_models(self, models_dict, X_val, y_val):
        '''
        多模型在验证集上的性能对比（选型用）

        注意：选型口径统一为验证集，最终测试集评估由调用方在选型后进行。
        并列 F1 时按 MODEL_CONFIGS 配置顺序取第一个（与训练侧 tie-break 一致，
        不受调用方 dict 顺序影响）。

        参数:
            models_dict: {model_type: model}
            X_val, y_val: 验证集数据

        返回:
            best_type, best_model, all_metrics
        '''
        print("\n" + "=" * 60)
        print("📊 验证集性能对比（选型）")
        print("=" * 60)

        all_metrics = {}
        print(f"{'模型':<15} {'准确率':>10} {'F1':>10} {'精确率':>10} {'召回率':>10}")
        print("-" * 55)

        # 按 MODEL_CONFIGS 顺序迭代，保证并列时 tie-break 确定性
        config_order = list(MODEL_CONFIGS)
        ordered_types = sorted(
            models_dict,
            key=lambda t: config_order.index(t) if t in config_order else len(config_order),
        )
        for model_type in ordered_types:
            model = models_dict[model_type]
            metrics = self.evaluate(model, X_val, y_val)
            all_metrics[model_type] = metrics
            print(
                f"{model_type:<15} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['f1']:>10.4f} "
                f"{metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f}"
            )

        best_type = max(all_metrics, key=lambda k: all_metrics[k]['f1'])
        best_model = models_dict[best_type]

        print("=" * 55)
        print(f"🏆 验证集最佳模型（选型）: '{best_type}' (F1={all_metrics[best_type]['f1']:.4f})")

        return best_type, best_model, all_metrics

    def check_consistency(self, models_dict, X, y):
        '''
        交付一致性校验：模型与评估数据的特征维数、类别空间必须一致

        用于防止 processed 数据与模型文件版本错配（如只重跑了其中一边）。

        参数:
            models_dict: {model_type: model}
            X, y: 评估数据（任意切分）
        '''
        problems = []
        n_features = X.shape[1]
        classes = set(np.unique(y).tolist())

        for model_type, model in models_dict.items():
            if getattr(model, 'n_features_in_', n_features) != n_features:
                problems.append(
                    f"  - {model_type}: 特征维数不符 "
                    f"(模型 {getattr(model, 'n_features_in_', '?')} vs 数据 {n_features})"
                )
            model_classes = set(getattr(model, 'classes_', np.array([])).tolist())
            if model_classes != classes:
                problems.append(
                    f"  - {model_type}: 类别空间不符 "
                    f"(模型 {sorted(model_classes)} vs 数据 {sorted(classes)})"
                )

        if problems:
            raise ValueError(
                "模型与评估数据不一致，请重新运行 train.py --save:\n"
                + "\n".join(problems)
            )
        print(f"✅ 一致性校验通过: {len(models_dict)} 个模型 × "
              f"特征维数 {n_features} × 类别数 {len(classes)}")

    def get_error_analysis(self, y_true, y_pred, texts=None):
        '''
        错误分析：找出被错误分类的样本
        '''
        errors = np.where(y_true != y_pred)[0]
        print(f"\n❌ 错误分析: 共 {len(errors)} / {len(y_true)} 个错误 "
              f"({100 * len(errors) / len(y_true):.1f}%)")

        if len(errors) == 0:
            print("  完美！没有错误分类 🎉")
            return errors

        print(f"\n前 10 个错误样本:")
        print(f"{'#':<4} {'真实':<25} {'预测':<25}")
        print("-" * 54)
        for idx in errors[:10]:
            true_name = self.target_names[y_true[idx]] if self.target_names else str(y_true[idx])
            pred_name = self.target_names[y_pred[idx]] if self.target_names else str(y_pred[idx])
            print(f"{idx:<4} {true_name:<25} {pred_name:<25}")

        return errors


if __name__ == '__main__':
    from src.data_loader import load_raw_data
    from src.preprocessor import TextPreprocessor
    from src.trainer import ModelTrainer
    from configs.config import DEFAULT_CATEGORIES

    print("=" * 60)
    print("测试 ModelEvaluator")
    print("=" * 60)

    df = load_raw_data(categories=DEFAULT_CATEGORIES)
    preprocessor = TextPreprocessor()
    data_dict = preprocessor.prepare_data(df)

    trainer = ModelTrainer()
    trainer.train_all(data_dict)

    evaluator = ModelEvaluator(target_names=preprocessor.target_names)

    # 选型用验证集，最终评估用测试集
    best_type, best_model, all_metrics = evaluator.compare_models(
        trainer.models,
        data_dict['X_val'],
        data_dict['y_val'],
    )

    evaluator.print_report(best_model, data_dict['X_test'], data_dict['y_test'])
    evaluator.print_confusion_matrix(best_model, data_dict['X_test'], data_dict['y_test'])

    print("\n" + "=" * 60)
    print("评估完成 ✅")
    print("=" * 60)