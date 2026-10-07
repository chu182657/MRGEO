"""
deepseek_client.py
==================
ChatAnywhere / OpenAI-compatible API 客户端封装。

注意：
这个文件名虽然叫 deepseek_client.py，但本版本不是只给 DeepSeek 用。
它会读取 config.py 中的：

    API_KEY
    API_BASE_URL
    MODEL_NAME
    MAX_RETRIES
    TIMEOUT
    MAX_OUTPUT_TOKENS
    TEMPERATURE

因此只要 config.py 里：

    MODEL_NAME = "gpt-4o-mini"

它就会通过 ChatAnywhere 调用 gpt-4o-mini。

为了兼容旧代码，本文件同时提供：

    OpenAICompatClient
    DeepSeekClient = OpenAICompatClient
    call_model(prompt)
    call_deepseek(prompt)
    call_chatanywhere(prompt)
    call_gemini(prompt)
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

from config import (
    API_BASE_URL,
    API_KEY,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MODEL_NAME,
    OUTPUT_DIR,
    TEMPERATURE,
)

try:
    from config import TIMEOUT
except Exception:
    TIMEOUT = 60.0


class OpenAICompatClient:
    """
    OpenAI 兼容 API 客户端封装。
    适配 ChatAnywhere 的 /v1/chat/completions 接口。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        api_base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or API_KEY
        self.model_name = model_name or MODEL_NAME
        self.max_retries = max_retries if max_retries is not None else MAX_RETRIES
        self.timeout = timeout if timeout is not None else TIMEOUT
        self.url = api_base_url or API_BASE_URL
        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("openai_compat_client")
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
                log_dir / "openai_compat_client.log",
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

        return logger

    @staticmethod
    def _ok(text: str) -> Dict[str, Any]:
        return {
            "success": True,
            "text": text,
            "error": None,
        }

    @staticmethod
    def _fail(error_msg: str) -> Dict[str, Any]:
        return {
            "success": False,
            "text": "",
            "error": error_msg,
        }

    def send_request(self, prompt: str) -> Dict[str, Any]:
        """
        发送单次请求。

        返回统一格式：

        {
            "success": True / False,
            "text": "...",
            "error": "..."
        }
        """

        if (
            not self.api_key
            or "请在环境变量中设置" in str(self.api_key)
            or str(self.api_key).startswith("YOUR_")
        ):
            error_msg = "API key 未配置，请在 config.py 或环境变量中设置 API_KEY。"
            self.logger.error(error_msg)
            return self._fail(error_msg)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }

        last_error = "未知错误"

        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(
                    "发送 OpenAI兼容请求，attempt=%s/%s, model=%s",
                    attempt,
                    self.max_retries,
                    self.model_name,
                )

                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()
                result = response.json()

                choices = result.get("choices") or []

                text = None
                if choices:
                    msg = choices[0].get("message") or {}
                    text = msg.get("content")

                if not text:
                    last_error = f"响应结构异常或无文本内容: {result}"
                    self.logger.error(last_error)
                else:
                    return self._ok(str(text))

            except requests.exceptions.Timeout:
                last_error = f"请求超时（>{self.timeout}s）"
                self.logger.warning(
                    "请求超时，attempt=%s/%s",
                    attempt,
                    self.max_retries,
                )

            except requests.exceptions.RequestException as exc:
                err_body = ""
                try:
                    if hasattr(exc, "response") and exc.response is not None:
                        err_body = f" | body={exc.response.text[:500]}"
                except Exception:
                    pass

                last_error = f"请求异常: {exc}{err_body}"
                self.logger.warning(
                    "请求失败，attempt=%s/%s，error=%s",
                    attempt,
                    self.max_retries,
                    last_error,
                )

            except Exception as exc:
                last_error = f"未预期异常: {exc}"
                self.logger.exception(
                    "未预期异常，attempt=%s/%s",
                    attempt,
                    self.max_retries,
                )

            if attempt < self.max_retries:
                time.sleep(min(2 * attempt, 5))

        self.logger.error("请求最终失败: %s", last_error)
        return self._fail(last_error)


# 兼容旧脚本：
# 如果旧代码写的是 from deepseek_client import DeepSeekClient，也不会报错。
DeepSeekClient = OpenAICompatClient


def call_model(prompt: str) -> str:
    """
    简化封装：
    成功返回文本，失败返回空字符串。
    """
    client = OpenAICompatClient()
    result = client.send_request(prompt)
    return result["text"] if result.get("success") else ""


def call_deepseek(prompt: str) -> str:
    """
    兼容旧代码。

    注意：
    实际调用哪个模型由 config.MODEL_NAME 决定，
    不一定是 DeepSeek。
    """
    return call_model(prompt)


def call_chatanywhere(prompt: str) -> str:
    """
    ChatAnywhere 通用调用函数。
    """
    return call_model(prompt)


def call_gemini(prompt: str) -> str:
    """
    兼容旧代码。

    注意：
    实际调用哪个模型由 config.MODEL_NAME 决定，
    不一定是 Gemini。
    """
    return call_model(prompt)


if __name__ == "__main__":
    test_prompt = "请用一句话回答：你能正常工作吗？"
    result = OpenAICompatClient().send_request(test_prompt)
    print(result)