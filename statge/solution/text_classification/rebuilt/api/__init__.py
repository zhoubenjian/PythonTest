'''rebuilt API 包：把 predict.py 的预测能力以 HTTP 服务形式交付。

约定：在 rebuilt 目录内运行（python -m uvicorn api.main:app），
与既有 5 个模块的"运行目录即包路径"保持一致，不引入 sys.path 注入。
'''
