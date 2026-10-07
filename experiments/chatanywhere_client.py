"""
chatanywhere_client.py
======================
ChatAnywhere OpenAI-compatible API 客户端。

适用于：
- gpt-4o-mini
- gpt-4o
- claude 系列中转模型
- gemini 系列中转模型
- 任何 ChatAnywhere 支持的 /v1/chat/completions 模型

读取 config.py 中：
    API_KEY
    API_BASE_URL
    MODEL_NAME
    MAX_RETRIES
    TIMEOUT
    OUTPUT_DIR
    REQUEST_INTERVAL
    MAX_OUTPUT_TOKENS
    TEMPERATURE
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

from config import API_KEY, API_BASE_URL, MODEL_NAME, MAX_RETRIES, OUTPUT_DIR

try:
    from config import TIMEOUT
except Exception:
    TIMEOUT = 60.0

try:
    from config import MAX_OUTPUT_TOKENS
except Exception:
    MAX_OUTPUT_TOKENS = 1024

try:
    from config import TEMPERATURE
except Exception:
    TEMPERATURE = 0.2


class ChatAnywhereClient:
    """ChatAnywhere 的 OpenAI-compatible chat/completions 客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or API_KEY
        self.model_name = model_name or MODEL_NAME
        self.max_retries = max_retries if max_retries is not None else MAX_RETRIES
        self.timeout = timeout if timeout is not None else TIMEOUT
        self.url = API_BASE_URL
        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("chatanywhere_client")
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        try:
            log_dir = OUTPUT_DIR / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / "chatanywhere_client.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

        return logger

    @staticmethod
    def _ok(text: str) -> Dict[str, Any]:
        return {"success": True, "text": text, "error": None}

    @staticmethod
    def _fail(error_msg: str) -> Dict[str, Any]:
        return {"success": False, "text": "", "error": error_msg}

    def send_request(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key or self.api_key.startswith("YOUR_"):
            return self._fail("API_KEY 未配置，请在 config.py 中设置 ChatAnywhere API Key。")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }

        last_error = "未知错误"

        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(
                    "发送 ChatAnywhere 请求，model=%s，attempt=%s/%s",
                    self.model_name,
                    attempt,
                    self.max_retries,
                )

                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                text = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                if not text:
                    last_error = f"响应结构异常或无文本内容: {result}"
                    self.logger.error(last_error)
                else:
                    return self._ok(str(text))

            except requests.exceptions.Timeout:
                last_error = f"请求超时（>{self.timeout}s）"
                self.logger.warning(
                    "ChatAnywhere 请求超时，attempt=%s/%s",
                    attempt,
                    self.max_retries,
                )
            except requests.exceptions.RequestException as exc:
                last_error = f"请求异常: {exc}"
                self.logger.warning(
                    "ChatAnywhere 请求失败，attempt=%s/%s，error=%s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except Exception as exc:
                last_error = f"未预期异常: {exc}"
                self.logger.exception(
                    "ChatAnywhere 未预期异常，attempt=%s/%s",
                    attempt,
                    self.max_retries,
                )

            time.sleep(min(2 * attempt, 8))

        return self._fail(last_error)


# 兼容旧代码：有些脚本可能调用 call_gemini(prompt)
def call_chatanywhere(prompt: str) -> str:
    result = ChatAnywhereClient().send_request(prompt)
    return result.get("text", "") if result.get("success") else ""


def call_gemini(prompt: str) -> str:
    """兼容旧的 run_generation.py。即使函数名叫 gemini，也会按 config.MODEL_NAME 调 ChatAnywhere。"""
    return call_chatanywhere(prompt)
