"""DeepSeek 模型的集中配置入口。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


class ModelConfigurationError(ValueError):
    """模型环境变量缺失或无效。"""


def _read_float(name: str, default: float, *, minimum: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ModelConfigurationError(
            f"{name} 必须是数字，当前值为 {raw_value!r}。请修改 .env。"
        ) from exc
    if value < minimum:
        raise ModelConfigurationError(
            f"{name} 必须大于或等于 {minimum}，当前值为 {value}。请修改 .env。"
        )
    return value


def _read_int(name: str, default: int, *, minimum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ModelConfigurationError(
            f"{name} 必须是整数，当前值为 {raw_value!r}。请修改 .env。"
        ) from exc
    if value < minimum:
        raise ModelConfigurationError(
            f"{name} 必须大于或等于 {minimum}，当前值为 {value}。请修改 .env。"
        )
    return value


def get_llm() -> ChatDeepSeek:
    """读取环境配置并创建统一的 DeepSeek 聊天模型实例。"""

    load_dotenv(ENV_FILE, override=False)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ModelConfigurationError(
            "未找到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env，"
            "再把自己的 DeepSeek API Key 填入 .env。"
        )

    model_name = os.getenv("MODEL_NAME", "deepseek-chat").strip()
    if not model_name:
        raise ModelConfigurationError(
            "MODEL_NAME 不能为空。请在 .env 中设置 MODEL_NAME=deepseek-chat。"
        )

    temperature = _read_float("MODEL_TEMPERATURE", 0.0, minimum=0.0)
    timeout = _read_float("MODEL_TIMEOUT_SECONDS", 45.0, minimum=0.1)
    max_retries = _read_int("MODEL_MAX_RETRIES", 2, minimum=0)

    return ChatDeepSeek(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )
