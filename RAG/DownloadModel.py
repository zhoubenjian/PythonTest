'''
从HuggingFace上手动下载指定模型 保存到本地指定目录


向量化模型：
    all-MiniLM-L6-v2
        轻量英文向量化模型，速度快，注意：对中文效果一般

    bge-small-zh-v1.5
        对中文友好
'''
import os

# 必须全部放在导入huggingface_hub之前！顺序不能乱
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 关键：禁用XET，解决401报错
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download


repo_id = "BAAI/bge-small-zh-v1.5"    # "sentence-transformers/all-MiniLM-L6-v2"
local_model_dir = r"../local_models/bge-small-zh-v1.5"    # r"../local_models/all-MiniLM-L6-v2"

snapshot_download(
    repo_id = repo_id,
    local_dir = local_model_dir,
)