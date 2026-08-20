from pathlib import Path
from unittest.mock import Mock

import pytest

import src.facade as facade_module
from src.artifacts import ArtifactError
from src.chat_submission import create_chat_attachment
from src.model import ModelConfigurationError
from src.schemas import ChatMessage, new_agent_result


def test_invoke_rejects_unavailable_stage() -> None:
    result = facade_module.invoke("V4", "测试")

    assert result["stage"] == "V4"
    assert result["error"] == (
        "当前版本仅支持 V0、V1、V2 和 V3，收到的阶段为 V4。"
    )


def test_invoke_routes_v0_without_attachment_using_legacy_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = new_agent_result("V0", text="V0 回复")
    invoke_v0 = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, "invoke_v0", invoke_v0)

    result = facade_module.invoke("V0", "测试")

    assert result is expected
    invoke_v0.assert_called_once_with("测试")


@pytest.mark.parametrize(
    ("stage", "function_name"),
    [
        ("V0", "invoke_v0"),
        ("V1", "invoke_v1"),
        ("V2", "invoke_v2"),
        ("V3", "invoke_v3"),
    ],
)
def test_invoke_exactly_forwards_attachment_for_v0_to_v3(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    function_name: str,
) -> None:
    attachment = create_chat_attachment(
        name="notes.txt",
        media_type="text/plain",
        data=b"attachment body",
    )
    expected = new_agent_result(stage, text=f"{stage} 回复")
    target = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, function_name, target)
    history: list[ChatMessage] = [
        {"role": "user", "content": "上一问"},
        {"role": "assistant", "content": "上一答"},
    ]

    result = facade_module.invoke(
        stage,
        "测试",
        history=history,
        attachment=attachment,
    )

    assert result is expected
    if stage == "V0":
        target.assert_called_once_with("测试", attachment=attachment)
    else:
        target.assert_called_once_with(
            "测试",
            history=history,
            attachment=attachment,
        )


def test_invoke_routes_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = new_agent_result("V1", text="V1 回复")
    received_history = None

    def fake_invoke_v1(message: str, *, history=None):
        nonlocal received_history
        received_history = history
        return expected

    monkeypatch.setattr(facade_module, "invoke_v1", fake_invoke_v1)
    history: list[ChatMessage] = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一个提示？"},
    ]

    result = facade_module.invoke("v1", "测试", history=history)

    assert result is expected
    assert received_history is history


def test_invoke_routes_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = new_agent_result("V2", text="V2 回复")
    received_history = None

    def fake_invoke_v2(message: str, *, history=None):
        nonlocal received_history
        received_history = history
        return expected

    monkeypatch.setattr(facade_module, "invoke_v2", fake_invoke_v2)
    history: list[ChatMessage] = [
        {"role": "user", "content": "请帮我整理错题"},
        {"role": "assistant", "content": "请先发来原题。"},
    ]

    result = facade_module.invoke("v2", "这是原题", history=history)

    assert result is expected
    assert received_history is history


def test_invoke_routes_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = new_agent_result("V3", text="V3 回复")
    received_history = None

    def fake_invoke_v3(message: str, *, history=None):
        nonlocal received_history
        received_history = history
        return expected

    monkeypatch.setattr(facade_module, "invoke_v3", fake_invoke_v3)
    history: list[ChatMessage] = [
        {"role": "user", "content": "three times 是什么线索？"},
        {"role": "assistant", "content": "它表示发生了几次？"},
    ]

    result = facade_module.invoke("v3", "为什么？", history=history)

    assert result is expected
    assert received_history is history


def test_chat_v4_delegates_to_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = new_agent_result(
        "V4",
        text="V4 回复",
        waiting_for="student_message",
    )
    received = None

    def fake_chat_v4(message: str, thread_id: str):
        nonlocal received
        received = (message, thread_id)
        return expected

    monkeypatch.setattr(facade_module, "_chat_v4", fake_chat_v4)

    result = facade_module.chat_v4("继续", "student-1")

    assert result is expected
    assert received == ("继续", "student-1")


def test_chat_v4_exactly_forwards_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attachment = create_chat_attachment(
        name="notes.md",
        media_type="text/markdown",
        data=b"# attachment body",
    )
    expected = new_agent_result(
        "V4",
        text="V4 回复",
        waiting_for="student_message",
    )
    chat_v4 = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, "_chat_v4", chat_v4)

    result = facade_module.chat_v4(
        "继续",
        "student-1",
        attachment=attachment,
    )

    assert result is expected
    chat_v4.assert_called_once_with(
        "继续",
        "student-1",
        attachment=attachment,
    )


def test_model_configuration_facade_never_adds_secret_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary: facade_module.ModelConfigurationSummary = {
        "provider": "moonshot",
        "configured_providers": ["deepseek", "moonshot"],
    }
    reader = Mock(return_value=summary)
    saver = Mock(return_value=summary)
    monkeypatch.setattr(facade_module, "_get_model_configuration_summary", reader)
    monkeypatch.setattr(facade_module, "_save_model_configuration", saver)

    assert facade_module.get_model_configuration() is summary
    assert facade_module.save_model_configuration("moonshot", "secret") is summary
    assert "secret" not in str(summary)
    reader.assert_called_once_with()
    saver.assert_called_once_with("moonshot", "secret")


def test_model_connection_uses_minimal_v0_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = new_agent_result("V0", text="连接成功")
    invoke_v0 = Mock(return_value=expected)
    monkeypatch.setattr(facade_module, "invoke_v0", invoke_v0)

    result = facade_module.test_model_connection()

    assert result is expected
    invoke_v0.assert_called_once_with("请只回复：连接成功")


def test_get_app_status_reports_ready_without_creating_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(facade_module.platform, "python_version", lambda: "3.14.3")
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status == {
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


@pytest.mark.parametrize("python_version", ["3.14.0", "3.14.3", "3.14.99"])
def test_get_app_status_accepts_any_python_3_14_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_version: str,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        facade_module.platform,
        "python_version",
        lambda: python_version,
    )
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status["ready"] is True
    assert status["runtime_ready"] is True
    assert status["model_ready"] is True
    assert status["errors"] == []


@pytest.mark.parametrize("python_version", ["3.13.9", "3.15.0", "unknown"])
def test_get_app_status_rejects_other_or_invalid_python_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    python_version: str,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        facade_module.platform,
        "python_version",
        lambda: python_version,
    )
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status["ready"] is False
    assert status["runtime_ready"] is False
    assert status["model_ready"] is True
    assert any("Python 3.14.x" in error for error in status["errors"])


def test_get_app_status_rejects_invalid_verified_python_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for relative_path in facade_module._REQUIRED_APP_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "invalid\n" if path.name == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(facade_module.platform, "python_version", lambda: "3.14.3")
    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        lambda: "deepseek",
    )

    status = facade_module.get_app_status()

    assert status["ready"] is False
    assert status["runtime_ready"] is False
    assert status["model_ready"] is True
    assert any(".python-version" in error for error in status["errors"])


def test_get_app_status_explains_missing_configuration_and_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".python-version").write_text("3.14.3\n", encoding="utf-8")
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(facade_module.platform, "python_version", lambda: "3.13.0")

    def reject_configuration() -> str:
        raise ModelConfigurationError("缺少测试 API Key。")

    monkeypatch.setattr(
        facade_module,
        "validate_model_configuration",
        reject_configuration,
    )

    status = facade_module.get_app_status()

    assert status["ready"] is False
    assert status["runtime_ready"] is False
    assert status["model_ready"] is False
    assert status["model_provider"] is None
    assert status["model_error"] == "缺少测试 API Key。"
    assert "缺少测试 API Key。" in status["errors"]
    assert "课程运行所需文件不完整。" in status["errors"]
    assert "课程运行所需文件不完整。" in status["runtime_errors"]
    assert "student/prompt.md" in status["missing_files"]
    assert any("课程要求 Python 3.14.x" in error for error in status["errors"])


def _configure_lesson_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    student_root = tmp_path / "student"
    prompt_path = student_root / "prompt.md"
    skill_path = student_root / "skill" / "sorting-out-mistakes" / "SKILL.md"
    prompt_path.parent.mkdir(parents=True)
    skill_path.parent.mkdir(parents=True)
    prompt_path.write_text("# 原 Prompt\n", encoding="utf-8")
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        "description: 整理错题。\n"
        "---\n\n"
        "# 原 Skill\n\n"
        "先观察线索。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(facade_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        facade_module,
        "_LESSON_ARTIFACTS",
        {
            "V1": facade_module._LessonArtifactSpec(
                stage="V1",
                label="测试 Prompt",
                path=prompt_path,
                kind="prompt",
            ),
            "V2": facade_module._LessonArtifactSpec(
                stage="V2",
                label="测试 Skill",
                path=skill_path,
                kind="skill",
            ),
        },
    )
    return prompt_path, skill_path


def test_lesson_artifact_save_returns_real_diff_and_atomic_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path, _ = _configure_lesson_artifacts(monkeypatch, tmp_path)
    before = facade_module.get_lesson_artifact("v1")

    assert before is not None
    change = facade_module.save_lesson_artifact(
        "V1",
        "# 新 Prompt\n\n一次只问一个问题。",
        expected_digest=before["digest"],
    )

    assert change["changed"] is True
    assert change["before"] == before
    assert change["after"]["content"] == "# 新 Prompt\n\n一次只问一个问题。"
    assert "-# 原 Prompt" in change["diff"]
    assert "+# 新 Prompt" in change["diff"]
    assert prompt_path.read_text(encoding="utf-8").endswith("\n")
    assert not list(prompt_path.parent.glob(".prompt.md.*.tmp"))


def test_lesson_artifact_rejects_invalid_skill_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, skill_path = _configure_lesson_artifacts(monkeypatch, tmp_path)
    original = skill_path.read_text(encoding="utf-8")
    before = facade_module.get_lesson_artifact("V2")

    assert before is not None
    with pytest.raises(ArtifactError, match="缺少 YAML frontmatter"):
        facade_module.save_lesson_artifact(
            "V2",
            "# 缺少 frontmatter",
            expected_digest=before["digest"],
        )

    assert skill_path.read_text(encoding="utf-8") == original
    assert not list(skill_path.parent.glob(".SKILL.md.*.tmp"))


def test_lesson_artifact_detects_external_edit_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path, _ = _configure_lesson_artifacts(monkeypatch, tmp_path)
    before = facade_module.get_lesson_artifact("V1")

    assert before is not None
    prompt_path.write_text("# 其他页面的新内容\n", encoding="utf-8")

    with pytest.raises(
        facade_module.ArtifactConflictError,
        match="已被其他页面修改",
    ):
        facade_module.save_lesson_artifact(
            "V1",
            "# 当前页面的内容",
            expected_digest=before["digest"],
        )

    assert prompt_path.read_text(encoding="utf-8") == "# 其他页面的新内容\n"


def test_v0_has_no_editable_lesson_artifact() -> None:
    assert facade_module.get_lesson_artifact("V0") is None

    with pytest.raises(ArtifactError, match="V0 没有需要保存"):
        facade_module.save_lesson_artifact(
            "V0",
            "不会保存",
            expected_digest="unused",
        )
