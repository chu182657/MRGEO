"""
config.py
=========
GEO 实验全局配置文件。
适用于 333 文件夹：ChatAnywhere / gpt-4o-mini / 23指标评测。
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

# 推荐：优先从环境变量读取
# PowerShell 设置方式：
# $env:API_KEY="你的 ChatAnywhere API Key"
#
# 如果你不用环境变量，就把下面占位符换成你的真实 key。
API_KEY = os.getenv("API_KEY", "")

# 当前 333 使用的评测模型
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ChatAnywhere OpenAI-compatible 接口
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://api.chatanywhere.tech/v1/chat/completions"
)

# ──────────────────────────────────────────────
# 实验方法列表：14 种 GEO 方法
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

# 请求失败最多重试 3 次
MAX_RETRIES = 3

# 当前评测脚本本身是串行，这里设 1 更稳
CONCURRENCY = 1

# Judge 请求较长，给足超时时间
TIMEOUT = 120.0

# 必须打开，否则可能只写 0 分不真正打分
ENABLE_JUDGE = True

# 关键：之前 0.5 太快，容易触发 403 / 风控 / 限流
# 建议先用 5 秒，稳定后可以再降到 3 秒
REQUEST_INTERVAL = 5.0

# 23指标 Judge 输出 JSON 很长，512 容易截断
MAX_OUTPUT_TOKENS = 2048

# 评测建议低温，分数更稳定
TEMPERATURE = 0.2

# ──────────────────────────────────────────────
# 启动校验
# ──────────────────────────────────────────────

# API clients validate credentials at request time; offline imports remain available.
