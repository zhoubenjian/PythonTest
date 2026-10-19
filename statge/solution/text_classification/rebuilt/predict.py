'''rebuilt 独立重写版：预测入口（新进程加载产物 → 单条/批量预测）

约束：不得 import 上级 src/ 的任何模块。

运行方式（在 rebuilt 目录内，需先跑通 fit.py 生成 artifacts）：
    & "D:\\ProgramData\\anaconda3\\envs\\ai_app\\python.exe" predict.py
'''
import os

# 与 fit.py 里同一路径；这里单独算一遍是为了让预测路径不必 import 训练模块
_HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(_HERE, 'artifacts')
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pkl')


def load_artifacts(artifact_dir=ARTIFACT_DIR):
    '''加载 vectorizer 与 model，返回 (vectorizer, model)。

    要求：文件不存在时给出明确错误，提示先运行 fit.py。
    '''
    raise NotImplementedError('TODO: 实现，用 joblib.load')


def predict_one(text, vectorizer, model, top_k=3):
    '''预测单条文本，返回 dict。

    返回的 dict 至少包含:
        label:      预测类别
        confidence: 该类别概率
        top_k:      [(类别, 概率), ...] 按概率降序取前 k 个
    '''
    raise NotImplementedError('TODO: 实现，注意要 transform 而不是 fit_transform')


def predict_batch(texts, vectorizer, model):
    '''批量预测，返回 list[dict]，每个元素与 predict_one 结构一致。'''
    raise NotImplementedError('TODO: 实现')


if __name__ == '__main__':
    vectorizer, model = load_artifacts()

    # 自测：三条文本，其中最后一条故意用模型没见过的领域，观察它会被分到哪一类、
    # 置信度是否明显低于前两条 —— 这是后面做错分分析要用的手感
    samples = [
        'The team scored three runs in the ninth inning to win the game.',
        'Computer graphics rendering uses polygon meshes and shaders.',
        'Quantum error correction requires fault-tolerant logical qubits.',
    ]
    for text in samples:
        print(predict_one(text, vectorizer, model))

    print(predict_batch(samples, vectorizer, model))
