from unittest.mock import Mock

import pytest

import src.model as model_module


@pytest.fixture(autouse=True)
def disable_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "load_dotenv", lambda *args, **kwargs: False)


def test_get_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(model_module.ModelConfigurationError, match="DEEPSEEK_API_KEY"):
        model_module.get_llm()


def test_get_llm_uses_centralized_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.2")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("MODEL_MAX_RETRIES", "3")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="deepseek-chat",
        temperature=0.2,
        timeout=30.0,
        max_retries=3,
        api_key="test-key",
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MODEL_TEMPERATURE", "warm", "MODEL_TEMPERATURE 必须是数字"),
        ("MODEL_TIMEOUT_SECONDS", "0", "MODEL_TIMEOUT_SECONDS 必须大于或等于"),
        ("MODEL_MAX_RETRIES", "1.5", "MODEL_MAX_RETRIES 必须是整数"),
    ],
)
def test_get_llm_rejects_invalid_numeric_settings(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(model_module.ModelConfigurationError, match=message):
        model_module.get_llm()
