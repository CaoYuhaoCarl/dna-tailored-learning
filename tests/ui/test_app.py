from pathlib import Path
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

import src.facade as facade_module
import src.model as model_module
import src.progress as progress_module
from src.schemas import new_agent_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ready_status() -> facade_module.AppStatus:
    return {
        "ready": True,
        "runtime_ready": True,
        "model_ready": True,
        "python_version": "3.14.3",
        "expected_python_version": "3.14.3",
        "model_provider": "deepseek",
        "missing_files": [],
        "runtime_errors": [],
        "model_error": None,
        "errors": [],
    }


def _unconfigured_status() -> facade_module.AppStatus:
    return {
        "ready": False,
        "runtime_ready": True,
        "model_ready": False,
        "python_version": "3.14.3",
        "expected_python_version": "3.14.3",
        "model_provider": None,
        "missing_files": [],
        "runtime_errors": [],
        "model_error": "缺少测试 API Key。",
        "errors": ["缺少测试 API Key。"],
    }


def _displayed_text(app: AppTest) -> str:
    element_groups = (
        app.markdown,
        app.caption,
        app.success,
        app.info,
        app.warning,
        app.error,
        app.code,
    )
    return "\n".join(
        str(element.value)
        for elements in element_groups
        for element in elements
    )


def test_home_loads_once_and_never_constructs_model(monkeypatch) -> None:
    progress = progress_module.default_progress()
    progress["completed_modules"] = ["v0", "v1"]
    load_progress = Mock(return_value=progress)
    get_app_status = Mock(return_value=_ready_status())
    get_model_configuration = Mock(
        return_value={
            "provider": "deepseek",
            "configured_providers": ["deepseek"],
        }
    )
    get_llm = Mock(side_effect=AssertionError("首页不能创建模型"))
    monkeypatch.setattr(progress_module, "load_progress", load_progress)
    monkeypatch.setattr(facade_module, "get_app_status", get_app_status)
    monkeypatch.setattr(
        facade_module,
        "get_model_configuration",
        get_model_configuration,
    )
    monkeypatch.setattr(model_module, "get_llm", get_llm)

    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()

    assert not app.exception
    assert app.title[0].value == "你的学习 Agent"
    assert app.success[0].value == (
        "课程文件和模型配置已就绪，当前模型供应商：DeepSeek。"
    )
    assert app.expander[0].proto.expanded is False
    assert app.session_state["progress"]["completed_modules"] == ["v0", "v1"]
    thread_id = app.session_state["thread_id"]
    assert thread_id.startswith("student-")
    load_progress.assert_called_once_with()
    get_llm.assert_not_called()

    app.run()

    assert not app.exception
    assert app.session_state["thread_id"] == thread_id
    load_progress.assert_called_once_with()
    get_llm.assert_not_called()


def test_home_shows_progress_recovery_and_configuration_steps(monkeypatch) -> None:
    monkeypatch.setattr(
        progress_module,
        "load_progress",
        Mock(side_effect=progress_module.ProgressDataError("测试损坏进度")),
    )
    monkeypatch.setattr(
        facade_module,
        "get_app_status",
        lambda: {
            "ready": False,
            "runtime_ready": False,
            "model_ready": False,
            "python_version": "3.14.3",
            "expected_python_version": "3.14.3",
            "model_provider": None,
            "missing_files": ["student/prompt.md"],
            "runtime_errors": ["课程运行所需文件不完整。"],
            "model_error": "缺少测试 API Key。",
            "errors": [
                "缺少测试 API Key。",
                "课程运行所需文件不完整。",
            ],
        },
    )
    monkeypatch.setattr(
        facade_module,
        "get_model_configuration",
        lambda: {"provider": "deepseek", "configured_providers": []},
    )

    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()

    assert not app.exception
    assert "测试损坏进度" in app.warning[0].value
    assert "运行环境还差一步" in app.error[0].value
    assert [expander.label for expander in app.expander] == [
        "模型配置",
        "家长修复步骤",
    ]
    assert app.expander[0].proto.expanded is True


def test_home_saves_password_key_and_tests_all_provider_option(
    monkeypatch,
) -> None:
    state = {"configured": False}
    secret = "ui-secret-must-not-be-rendered"

    def get_status() -> facade_module.AppStatus:
        if state["configured"]:
            status = _ready_status()
            status["model_provider"] = "gemini"
            return status
        return _unconfigured_status()

    def get_configuration() -> facade_module.ModelConfigurationSummary:
        return {
            "provider": "gemini" if state["configured"] else "deepseek",
            "configured_providers": ["gemini"] if state["configured"] else [],
        }

    save_configuration = Mock()

    def save(provider: str, api_key: str):
        save_configuration(provider, api_key)
        state["configured"] = True
        return get_configuration()

    test_connection = Mock(return_value=new_agent_result("V0", text="连接成功"))
    monkeypatch.setattr(facade_module, "get_app_status", get_status)
    monkeypatch.setattr(facade_module, "get_model_configuration", get_configuration)
    monkeypatch.setattr(facade_module, "save_model_configuration", save)
    monkeypatch.setattr(facade_module, "test_model_connection", test_connection)

    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()

    assert app.expander[0].label == "模型配置"
    assert app.selectbox[0].options == ["DeepSeek", "Kimi", "Gemini"]
    assert app.text_input[0].proto.type == app.text_input[0].proto.PASSWORD

    app.selectbox[0].select("Gemini")
    app.text_input[0].input(secret)
    app.button[0].click()
    app.run()

    assert not app.exception
    save_configuration.assert_called_once_with("gemini", secret)
    test_connection.assert_called_once_with()
    assert any("连接测试成功" in item.value for item in app.success)
    assert secret not in _displayed_text(app)


def test_home_keeps_saved_configuration_when_connection_test_fails(
    monkeypatch,
) -> None:
    save_configuration = Mock(
        return_value={
            "provider": "moonshot",
            "configured_providers": ["moonshot"],
        }
    )
    test_connection = Mock(
        return_value=new_agent_result(
            "V0",
            error=(
                "模型调用失败。"
                "请检查网络、API Key、账户余额和供应商配置。"
            ),
        )
    )
    monkeypatch.setattr(facade_module, "get_app_status", lambda: _unconfigured_status())
    monkeypatch.setattr(
        facade_module,
        "get_model_configuration",
        lambda: {"provider": "moonshot", "configured_providers": ["moonshot"]},
    )
    monkeypatch.setattr(
        facade_module,
        "save_model_configuration",
        save_configuration,
    )
    monkeypatch.setattr(facade_module, "test_model_connection", test_connection)

    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()
    app.selectbox[0].select("Kimi")
    app.text_input[0].input("")
    app.button[0].click()
    app.run()

    assert not app.exception
    save_configuration.assert_called_once_with("moonshot", "")
    assert any(
        "配置已保存，但连接测试失败" in item.value
        for item in app.error
    )
