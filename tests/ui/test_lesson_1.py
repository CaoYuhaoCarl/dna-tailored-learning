from pathlib import Path
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

import src.facade as facade_module
import src.progress as progress_module
from src.schemas import new_agent_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = PROJECT_ROOT / "app_pages" / "lesson_1.py"


def _artifact(stage: str, content: str, digest: str) -> facade_module.LessonArtifact:
    if stage == "V1":
        path = "student/prompt.md"
        label = "苏格拉底教练 Prompt"
    else:
        path = "student/skill/sorting-out-mistakes/SKILL.md"
        label = "整理错题 Skill"
    return {
        "stage": stage,
        "label": label,
        "path": path,
        "content": content,
        "digest": digest,
    }


def _configure_page(monkeypatch):
    artifacts = {
        "V1": _artifact("V1", "# 原 Prompt", "prompt-old"),
        "V2": _artifact(
            "V2",
            "---\nname: sorting-out-mistakes\n"
            "description: 整理错题。\n---\n\n# 原 Skill",
            "skill-old",
        ),
    }
    get_artifact = Mock(side_effect=lambda stage: artifacts[stage].copy())
    invoke = Mock(return_value=new_agent_result("V1", text="先观察哪个线索？"))
    save_artifact = Mock()

    def save_side_effect(stage, content, *, expected_digest):
        before = artifacts[stage].copy()
        after = before.copy()
        after["content"] = content.strip()
        after["digest"] = f"{stage.lower()}-new"
        artifacts[stage] = after
        return {
            "before": before,
            "after": after.copy(),
            "changed": before["content"] != after["content"],
            "diff": "-旧内容\n+新内容",
        }

    save_artifact.side_effect = save_side_effect
    load_progress = Mock(return_value=progress_module.default_progress())
    save_progress = Mock(side_effect=lambda progress: progress)

    monkeypatch.setattr(facade_module, "get_lesson_artifact", get_artifact)
    monkeypatch.setattr(facade_module, "save_lesson_artifact", save_artifact)
    monkeypatch.setattr(facade_module, "invoke", invoke)
    monkeypatch.setattr(progress_module, "load_progress", load_progress)
    monkeypatch.setattr(progress_module, "save_progress", save_progress)
    return get_artifact, save_artifact, invoke, load_progress, save_progress


def _widget_by_label(elements, label: str):
    return next(element for element in elements if element.label == label)


def test_lesson_1_non_chat_actions_never_invoke_model(monkeypatch) -> None:
    _, save_artifact, invoke, load_progress, _ = _configure_page(monkeypatch)

    app = AppTest.from_file(PAGE_PATH).run()

    assert not app.exception
    assert not app.main.title
    assert app.sidebar.header[0].value == "第一课"
    assert not app.main.segmented_control
    assert len(app.sidebar.segmented_control) == 1
    assert not app.main.expander
    assert app.sidebar.status[0].label == "比较三个助手的回答"
    assert len(app.main.chat_message) == 1
    assert not app.sidebar.chat_message
    assert app.chat_input[0].placeholder == "和原始普通AI聊天"
    invoke.assert_not_called()
    save_artifact.assert_not_called()
    load_progress.assert_called_once_with()

    app.run()
    app.segmented_control[0].set_value("V1").run()
    _widget_by_label(app.text_area, "写给教学助手的说明书").set_value(
        "# 修改后的 Prompt"
    ).run()

    assert not app.exception
    invoke.assert_not_called()
    save_artifact.assert_not_called()
    load_progress.assert_called_once_with()


def test_lesson_1_saves_training_separately_then_chats_once(monkeypatch) -> None:
    _, save_artifact, invoke, _, save_progress = _configure_page(monkeypatch)
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V1").run()
    _widget_by_label(app.text_area, "写给教学助手的说明书").set_value(
        "# 新 Prompt\n\n一次只问一个问题。"
    )

    _widget_by_label(app.button, "保存").click().run()

    assert not app.exception
    save_artifact.assert_called_once_with(
        "V1",
        "# 新 Prompt\n\n一次只问一个问题。",
        expected_digest="prompt-old",
    )
    invoke.assert_not_called()
    save_progress.assert_not_called()

    app.chat_input[0].set_value(
        "She ___ (go) to Shanghai four times."
    ).run()

    assert not app.exception
    save_artifact.assert_called_once()
    invoke.assert_called_once_with(
        "V1",
        "She ___ (go) to Shanghai four times.",
        history=[],
    )
    save_progress.assert_called_once()
    assert app.session_state["lesson1_histories"]["V1"] == [
        {
            "role": "user",
            "content": "She ___ (go) to Shanghai four times.",
        },
        {"role": "assistant", "content": "先观察哪个线索？"},
    ]
    assert "v1" in app.session_state["progress"]["completed_modules"]

    app.run()

    save_artifact.assert_called_once()
    invoke.assert_called_once()
    save_progress.assert_called_once()


def test_lesson_1_save_error_never_sends_a_chat_message(monkeypatch) -> None:
    _, save_artifact, invoke, _, _ = _configure_page(monkeypatch)
    save_artifact.side_effect = facade_module.ArtifactConflictError(
        "文件已被其他页面修改。"
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V2").run()

    _widget_by_label(app.button, "保存").click().run()

    assert not app.exception
    save_artifact.assert_called_once()
    invoke.assert_not_called()
    assert "已被其他页面修改" in app.error[0].value


def test_lesson_1_v2_keeps_separate_history_and_explains_tool_calls(
    monkeypatch,
) -> None:
    _, save_artifact, invoke, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result(
        "V2",
        text="已按 Skill 整理。",
        tool_calls=[
            {
                "name": "load_skill",
                "args": {"skill_name": "sorting-out-mistakes"},
            }
        ],
    )
    app = AppTest.from_file(PAGE_PATH).run()
    app.segmented_control[0].set_value("V2").run()

    app.chat_input[0].set_value("请整理错题").run()

    assert not app.exception
    save_artifact.assert_not_called()
    invoke.assert_called_once_with("V2", "请整理错题", history=[])
    assert app.session_state["lesson1_histories"]["V1"] == []
    assert app.session_state["lesson1_histories"]["V2"][-1] == {
        "role": "assistant",
        "content": "已按 Skill 整理。",
    }
    assert app.session_state["lesson1_last_results"]["V2"]["tool_calls"][0][
        "name"
    ] == "load_skill"
    assert any(
        "读取了整理错题的方法" in markdown.value for markdown in app.markdown
    )
    assert any("load_skill" in markdown.value for markdown in app.markdown)


def test_lesson_1_v0_chat_runs_without_saving_markdown(monkeypatch) -> None:
    _, save_artifact, invoke, _, _ = _configure_page(monkeypatch)
    invoke.return_value = new_agent_result("V0", text="普通回答")
    app = AppTest.from_file(PAGE_PATH).run()

    app.chat_input[0].set_value("你好").run()

    assert not app.exception
    save_artifact.assert_not_called()
    invoke.assert_called_once_with("V0", "你好", history=[])
    assert app.session_state["lesson1_histories"]["V0"][-1] == {
        "role": "assistant",
        "content": "普通回答",
    }
    chat_rows = [
        node
        for node in app.main.children.values()
        if node.type == "flex_container"
    ]
    alignments = [
        row.proto.flex_container.Align.Name(row.proto.flex_container.align)
        for row in chat_rows
    ]
    assert alignments == ["ALIGN_START", "ALIGN_END", "ALIGN_START"]
    assert [message.name for message in app.main.chat_message] == [
        "assistant",
        "user",
        "assistant",
    ]


def test_lesson_1_compares_each_stages_latest_reply(monkeypatch) -> None:
    _, _, invoke, _, _ = _configure_page(monkeypatch)
    invoke.side_effect = [
        new_agent_result("V0", text="V0 的回答"),
        new_agent_result("V1", text="V1 的回答"),
    ]
    app = AppTest.from_file(PAGE_PATH).run()
    app.chat_input[0].set_value("同一道题").run()
    app.segmented_control[0].set_value("V1").run()
    app.chat_input[0].set_value("同一道题").run()

    replies = [markdown.value for markdown in app.markdown]
    assert any("V0 的回答" in reply for reply in replies)
    assert any("V1 的回答" in reply for reply in replies)
