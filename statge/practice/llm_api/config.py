"""公共配置：读根目录 .env，暴露 API_KEY / API_BASE / MODEL。

本文件是基础设施，直接可用，不需要你改。
注意：密钥只从环境变量读，不写进代码，也不打印出来。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent  # statge/practice/llm_api -> 仓库根

load_dotenv(PROJECT_ROOT / '.env')

API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
API_BASE = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/')
MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
TIMEOUT = float(os.getenv('DEEPSEEK_TIMEOUT', '60'))

if not API_KEY:
    raise RuntimeError(
        f'未找到 DEEPSEEK_API_KEY，请确认 {PROJECT_ROOT / ".env"} 中存在该变量'
    )
