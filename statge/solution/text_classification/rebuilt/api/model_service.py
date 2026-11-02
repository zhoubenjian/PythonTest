'''rebuilt API：模型服务层（加载产物 + 类型转换 + 日志）。

分工与 predict.py 一致：训练侧定，预测侧只执行——这里不挑文件、
不做选型，只加载固定产物。预测逻辑直接复用 predict.py 的
predict_one / predict_batch，不复制实现（两份实现迟早漂移）。

注意：本模块 import 上级的 predict，要求运行目录是 rebuilt/
（uvicorn 会把启动时的工作目录放进 sys.path）。
'''
import logging
import time

from predict import load_artifacts, predict_batch, predict_one

logger = logging.getLogger('api.service')


class ModelService:
    '''进程内单例：启动时加载一次产物，之后所有请求共用。

    加载失败直接抛 FileNotFoundError（早失败），让 uvicorn 启动即报错，
    而不是等第一个请求进来才炸——错误离原因越近越好查。
    '''

    def __init__(self):
        t0 = time.perf_counter()
        self.vectorizer, self.model = load_artifacts()
        self.load_seconds = time.perf_counter() - t0
        self.n_features = self.vectorizer.transform(['']).shape[1]
        self.classes = [str(c) for c in self.model.classes_]
        logger.info(
            'artifacts loaded in %.3fs: n_features=%d classes=%s',
            self.load_seconds, self.n_features, self.classes,
        )

    def predict(self, text: str, top_k: int) -> dict:
        '''单条预测。top_k 先收敛到类别数，再交给 predict_one。'''
        result = predict_one(text, self.vectorizer, self.model, top_k=min(top_k, len(self.classes)))
        return self._to_plain(result)

    def predict_texts(self, texts: list[str], top_k: int) -> list[dict]:
        '''批量预测。顺序与输入一一对应（predict_batch 的契约）。'''
        results = predict_batch(texts, self.vectorizer, self.model, top_k=min(top_k, len(self.classes)))
        return [self._to_plain(r) for r in results]

    def _to_plain(self, result: dict) -> dict:
        '''numpy 标量/类型转原生 Python 类型，保证 pydantic 与 JSON 序列化不出幺蛾子。'''
        return {
            'label': str(result['label']),
            'confidence': float(result['confidence']),
            'top_k': [(str(label), float(p)) for label, p in result['top_k']],
        }


def timed(service: ModelService, fn, *args, **kwargs):
    '''记录一次预测耗时。API 的性能观测从最简单的时间日志起步，不够再上中间件。'''
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    logger.info('predicted in %.1fms', (time.perf_counter() - t0) * 1000)
    return out
