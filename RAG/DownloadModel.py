'''
从HuggingFace上手动下载模型，保存到本地指定目录。
'''
import os

# 必须全部放在导入huggingface_hub之前！顺序不能乱
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"   # 关键：禁用XET，解决401报错

from huggingface_hub import snapshot_download

repo_id = "sentence-transformers/all-MiniLM-L6-v2"
local_model_dir = r"D:\PythonProject\Github\PythonTest\local_models\all-MiniLM-L6-v2"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_model_dir,
)