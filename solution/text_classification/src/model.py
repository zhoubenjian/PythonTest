'''
模型工厂模块
根据配置动态创建 sklearn 分类模型
'''
import os
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import MODEL_CONFIGS


def get_model(model_type):
    '''
    根据模型类型创建 sklearn 模型实例
    
    参数:
        model_type: 模型 key (如 'naive_bayes', 'logistic', 'svm', 'forest')
    
    返回:
        sklearn estimator
    '''
    if model_type not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(
            f"未知模型类型: '{model_type}'\n"
            f"可用模型: {available}"
        )

    config = MODEL_CONFIGS[model_type]
    model_name = config['name']
    model_params = config['params']

    from sklearn import naive_bayes, linear_model, svm, ensemble

    cls = None
    for module in [naive_bayes, linear_model, svm, ensemble]:
        if hasattr(module, model_name):
            cls = getattr(module, model_name)
            break

    if cls is None:
        raise ImportError(f"无法找到模型类: {model_name}")

    model = cls(**model_params)
    print(f"🔧 创建模型: {model_type} → {model_name}({model_params})")
    return model


def save_model(model, filepath):
    '''
    保存模型到文件
    '''
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"💾 模型已保存: {filepath}")
    return filepath


def load_model(filepath):
    '''
    从文件加载模型
    '''
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    model = joblib.load(filepath)
    print(f"📂 模型已加载: {filepath}")
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("测试 model.py")
    print("=" * 60)

    for model_type in MODEL_CONFIGS:
        print(f"\n--- 创建 '{model_type}' ---")
        model = get_model(model_type)
        print(f"  类型: {type(model).__name__}")
        print(f"  参数量: {len(model.get_params())}")

    print("\n" + "=" * 60)
    print("所有模型创建成功 ✅")
    print("=" * 60)