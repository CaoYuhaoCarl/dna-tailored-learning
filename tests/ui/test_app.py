from pathlib import Path
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

import src.facade as facade_module
import src.model as model_module
import src.progress as progress_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ready_status() -> facade_module.AppStatus:
    return {
        "ready": True,
        "python_version": "3.14.3",
        "expected_python_version": "3.14.3",
        "model_provider": "deepseek",
        "missing_files": [],
        "errors": [],
    }


def test_home_loads_once_and_never_constructs_model(monkeypatch) -> None:
    progress = progress_module.default_progress()
    progress["completed_modules"] = ["v0", "v1"]
    load_progress = Mock(return_value=progress)
    get_app_status = Mock(return_value=_ready_status())
    get_llm = Mock(side_effect=AssertionError("首页不能创建模型"))
    monkeypatch.setattr(progress_module, "load_progress", load_progress)
    monkeypatch.setattr(facade_module, "get_app_status", get_app_status)
    monkeypatch.setattr(model_module, "get_llm", get_llm)

    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()

    assert not app.exception
    assert app.title[0].value == "你的学习 Agent"
    assert app.success[0].value == (
        "运行环境已就绪，当前模型供应商：deepseek。"
    )
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
            "python_version": "3.14.3",
            "expected_python_version": "3.14.3",
            "model_provider": None,
            "missing_files": ["student/prompt.md"],
            "errors": [
                "缺少测试 API Key。",
                "课程运行所需文件不完整。",
            ],
        },
    )

    app = AppTest.from_file(PROJECT_ROOT / "app.py").run()

    assert not app.exception
    assert "测试损坏进度" in app.warning[0].value
    assert "运行环境还差一步" in app.error[0].value
    assert app.expander[0].label == "家长修复步骤"
