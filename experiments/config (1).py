"""
config.py
=========
GEO 实验全局配置文件（Claude Haiku 4.5 · chatanywhere 版本）
"""

from pathlib import Path
import os

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "samples.json"
OUTPUT_DIR = Path(os.getenv("MRGEO_OUTPUT_DIR", str(BASE_DIR / "outputs")))

# ──────────────────────────────────────────────
# API 配置
# ──────────────────────────────────────────────

# 第 28 行：把引号里换成你的 chatanywhere API Key
API_KEY = os.getenv("API_KEY", "")

# ✅ 已修正为 chatanywhere 实际支持的模型名
MODEL_NAME = os.getenv("MODEL_NAME", "claude-haiku-4-5-20251001")

# chatanywhere 固定接口地址，无需修改
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.chatanywhere.tech/v1")

# ──────────────────────────────────────────────
# 实验方法列表
# ──────────────────────────────────────────────

METHOD_LIST = [
    "uniqueness",
    "simplification",
    "authority",
    "fluency",
    "terminology",
    "reputation",
    "citation",
    "statistics",
    "autogeo",
    "intent_geo",
    "multimodal",
    "multi_agent",
    "rag_based",
    "m2geo",
]

# ──────────────────────────────────────────────
# 运行参数
# ──────────────────────────────────────────────

MAX_RETRIES = 3
CONCURRENCY = 2          # chatanywhere 建议不超过 3
TIMEOUT = 90.0
ENABLE_JUDGE = os.getenv("ENABLE_JUDGE", "false").lower() == "true"
REQUEST_INTERVAL = 0.8

# 成本控制
MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.2

# ──────────────────────────────────────────────
# 启动校验
# ──────────────────────────────────────────────

# API clients validate credentials at request time; offline imports remain available.
