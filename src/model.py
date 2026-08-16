"""所有学习 Agent 共用的模型配置入口。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
MODEL_PROVIDER_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ModelConfigurationError(ValueError):
    """模型环境变量缺失或无效。"""


def get_model_name(llm: BaseChatModel) -> str:
    """返回不同 LangChain 模型适配器使用的模型名称。"""

    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    if not isinstance(model_name, str) or not model_name:
        raise ModelConfigurationError("当前模型适配器未提供可显示的模型名称。")
    return model_name


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


def validate_model_configuration() -> str:
    """校验模型供应商和 API Key，但不创建模型实例。"""

    load_dotenv(ENV_FILE, override=False)

    provider = os.getenv("MODEL_PROVIDER", "deepseek").strip().lower()
    key_name = MODEL_PROVIDER_KEYS.get(provider)
    if key_name is None:
        raise ModelConfigurationError(
            "MODEL_PROVIDER 必须是 deepseek、moonshot 或 gemini，"
            f"当前值为 {provider!r}。请修改 .env。"
        )

    if not os.getenv(key_name, "").strip():
        raise ModelConfigurationError(
            f"MODEL_PROVIDER={provider}，但未找到 {key_name}。"
            "请复制 .env.example 为 .env，再填写所选供应商的 API Key。"
        )

    return provider


def get_llm() -> BaseChatModel:
    """根据 MODEL_PROVIDER 创建所有 V0-V4 共用的聊天模型实例。"""

    provider = validate_model_configuration()
    api_key = os.environ[MODEL_PROVIDER_KEYS[provider]].strip()

    timeout = _read_float("MODEL_TIMEOUT_SECONDS", 45.0, minimum=0.1)
    max_retries = _read_int("MODEL_MAX_RETRIES", 2, minimum=0)

    if provider == "deepseek":
        temperature = _read_float("MODEL_TEMPERATURE", 0.0, minimum=0.0)
        return ChatDeepSeek(
            model="deepseek-chat",
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
        )

    if provider == "moonshot":
        return ChatOpenAI(
            model="kimi-k2.6",
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            timeout=timeout,
            max_retries=max_retries,
            use_responses_api=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
