from unittest.mock import Mock

import pytest

import src.model as model_module


_MODEL_ENVIRONMENT_VARIABLES = (
    "MODEL_PROVIDER",
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "GEMINI_API_KEY",
    "MODEL_NAME",
    "MODEL_TEMPERATURE",
    "MODEL_TIMEOUT_SECONDS",
    "MODEL_MAX_RETRIES",
)


@pytest.fixture(autouse=True)
def isolate_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "load_dotenv", lambda *args, **kwargs: False)
    for name in _MODEL_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("moonshot", "MOONSHOT_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ],
)
def test_get_llm_requires_selected_provider_key(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_name: str,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", provider)

    with pytest.raises(model_module.ModelConfigurationError, match=key_name):
        model_module.get_llm()


def test_get_llm_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "unknown")

    with pytest.raises(
        model_module.ModelConfigurationError,
        match="deepseek、moonshot 或 gemini",
    ):
        model_module.get_llm()


def test_get_llm_defaults_to_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="deepseek-v4-flash",
        temperature=0.0,
        timeout=45.0,
        max_retries=2,
        api_key="deepseek-key",
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_get_llm_does_not_override_running_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    loader = Mock(return_value=False)
    monkeypatch.setattr(model_module, "load_dotenv", loader)
    monkeypatch.setattr(model_module, "ChatDeepSeek", Mock(return_value=object()))

    model_module.get_llm()

    loader.assert_called_once_with(model_module.ENV_FILE, override=False)


def test_validate_model_configuration_does_not_construct_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    constructor = Mock()
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    provider = model_module.validate_model_configuration()

    assert provider == "deepseek"
    constructor.assert_not_called()


def test_get_llm_selects_moonshot_and_only_requires_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "moonshot")
    monkeypatch.setenv("MOONSHOT_API_KEY", "moonshot-key")
    monkeypatch.setenv("MODEL_NAME", "ignored-model")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatOpenAI", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="kimi-k2.6",
        api_key="moonshot-key",
        base_url="https://api.moonshot.cn/v1",
        timeout=45.0,
        max_retries=2,
        use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
    )


def test_get_llm_selects_gemini_and_only_requires_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "GEMINI")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("MODEL_NAME", "ignored-model")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatGoogleGenerativeAI", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="gemini-3.6-flash",
        api_key="gemini-key",
        timeout=45.0,
        max_retries=2,
    )


@pytest.mark.parametrize(
    ("provider", "key_name", "expected_name"),
    [
        ("deepseek", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
        ("moonshot", "MOONSHOT_API_KEY", "kimi-k2.6"),
        ("gemini", "GEMINI_API_KEY", "gemini-3.6-flash"),
    ],
)
def test_get_model_name_supports_all_providers(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_name: str,
    expected_name: str,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    monkeypatch.setenv(key_name, "test-key")

    llm = model_module.get_llm()

    assert model_module.get_model_name(llm) == expected_name


def test_get_llm_uses_shared_timeout_and_retry_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.2")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("MODEL_MAX_RETRIES", "3")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(model_module, "ChatDeepSeek", constructor)

    model_module.get_llm()

    constructor.assert_called_once_with(
        model="deepseek-v4-flash",
        temperature=0.2,
        timeout=30.0,
        max_retries=3,
        api_key="test-key",
        extra_body={"thinking": {"type": "disabled"}},
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
