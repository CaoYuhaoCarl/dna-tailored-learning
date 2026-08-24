from base64 import b64decode
from pathlib import Path
from unittest.mock import Mock

from streamlit.proto.ChatInput_pb2 import ChatInput as ChatInputProto
from streamlit.testing.v1 import AppTest

import src.chat_submission as chat_submission_module
import src.facade as facade_module
from src.chat_submission import (
    ACCEPTED_FILE_TYPES,
    MAX_ATTACHMENT_BYTES,
    ChatAttachment,
    ChatSubmission,
    ChatSubmissionError,
)
from src.schemas import new_agent_result
from src.skill_pages import new_english_quest_session


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = PROJECT_ROOT / "app_pages" / "english_quest.py"
_TINY_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


def _button_by_label(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _image_submission(*, text: str = "") -> ChatSubmission:
    return ChatSubmission(
        text=text,
        attachment=ChatAttachment(
            name="question.png",
            media_type="image/png",
            data=_TINY_PNG,
        ),
    )


def test_english_quest_starts_and_continues_with_full_history(monkeypatch) -> None:
    invoke = Mock(
        side_effect=[
            new_agent_result(
                "V2",
                text=(
                    "🕵️ 任务：找出消失的英语课本\n"
                    "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP\n\n"
                    "I ___ this book three times.\n"
                    "A. read  B. have read  C. am reading\n\n"
                    "你选择哪一个？"
                ),
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_name": "english-quest"},
                    }
                ],
            ),
            new_agent_result(
                "V2",
                text=(
                    "线索正确。\n"
                    "关卡：2/5 生命：❤️❤️❤️ 经验：10 XP\n\n"
                    "下一条线索：She ___ already finished.\n\n"
                    "填什么？"
                ),
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_name": "english-quest"},
                    }
                ],
            ),
        ]
    )
    monkeypatch.setattr(facade_module, "invoke", invoke)

    app = AppTest.from_file(PAGE_PATH).run()

    assert not app.exception
    assert app.title[0].value == "英语剧情闯关"
    assert [metric.value for metric in app.metric] == ["0/5", "❤️❤️❤️", "0 XP"]
    assert any("scripts/quest_state.py" in item.value for item in app.markdown)
    _button_by_label(app, "开始任务").click().run()

    assert not app.exception
    invoke.assert_called_once_with(
        "V2",
        "我们玩一个侦探破案闯关游戏练英语现在完成时。",
        history=[],
    )
    assert [metric.value for metric in app.metric] == ["1/5", "❤️❤️❤️", "0 XP"]
    assert app.chat_input[0].proto.accept_file == ChatInputProto.SINGLE
    assert list(app.chat_input[0].proto.file_type) == [
        f".{extension}" for extension in ACCEPTED_FILE_TYPES
    ]
    assert app.chat_input[0].proto.max_upload_size_mb == (
        MAX_ATTACHMENT_BYTES // (1024 * 1024)
    )
    opening_history = [
        item.copy()
        for item in app.session_state["english_quest_session"]["history"]
    ]

    app.run()
    app.chat_input[0].set_value("B").run()

    assert not app.exception
    assert invoke.call_count == 2
    invoke.assert_called_with("V2", "B", history=opening_history)
    assert [metric.value for metric in app.metric] == ["2/5", "❤️❤️❤️", "10 XP"]
    assert app.session_state["english_quest_session"]["history"][-2:] == [
        {"role": "user", "content": "B"},
        {
            "role": "assistant",
            "content": (
                "线索正确。\n"
                "关卡：2/5 生命：❤️❤️❤️ 经验：10 XP\n\n"
                "下一条线索：She ___ already finished.\n\n"
                "填什么？"
            ),
        },
    ]


def test_english_quest_does_not_append_failed_turn(monkeypatch) -> None:
    invoke = Mock(return_value=new_agent_result("V2", error="测试模型不可用"))
    monkeypatch.setattr(facade_module, "invoke", invoke)
    app = AppTest.from_file(PAGE_PATH).run()

    _button_by_label(app, "开始任务").click().run()

    assert not app.exception
    assert app.session_state["english_quest_session"]["history"] == []
    assert app.session_state["english_quest_last_error"] == "测试模型不可用"
    assert any("测试模型不可用" in item.value for item in app.error)


def test_completed_quest_can_handoff_mistakes_to_next_skill(monkeypatch) -> None:
    invoke = Mock(
        return_value=new_agent_result(
            "V2",
            text="我会根据本局历史整理错题。",
            tool_calls=[
                {
                    "name": "load_skill",
                    "args": {"skill_name": "sorting-out-mistakes"},
                }
            ],
        )
    )
    monkeypatch.setattr(facade_module, "invoke", invoke)
    app = AppTest.from_file(PAGE_PATH).run()
    completed_history = [
        {"role": "user", "content": "开始游戏"},
        {
            "role": "assistant",
            "content": "任务报告\n最终 XP：40\n剩余生命：2",
        },
    ]
    app.session_state["english_quest_session"] = new_english_quest_session(
        history=completed_history,
    )

    app.run()
    _button_by_label(app, "整理本局错题").click().run()

    assert not app.exception
    invoke.assert_called_once_with(
        "V2",
        "把刚才答错的题整理进错题本。",
        history=completed_history,
    )
    assert any("sorting-out-mistakes" in item.value for item in app.success)


def test_english_quest_forwards_attachment_and_keeps_history_text_only(
    monkeypatch,
) -> None:
    opening_history = [
        {"role": "user", "content": "开始游戏"},
        {
            "role": "assistant",
            "content": "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP\n请看下一条线索。",
        },
    ]
    submission = _image_submission(text="请看图里的英语题")
    parser = Mock(return_value=submission)
    invoke = Mock(
        return_value=new_agent_result(
            "V2",
            text="关卡：2/5 生命：❤️❤️❤️ 经验：10 XP\n图片线索已识别。",
        )
    )
    monkeypatch.setattr(chat_submission_module, "parse_chat_submission", parser)
    monkeypatch.setattr(facade_module, "invoke", invoke)
    app = AppTest.from_file(PAGE_PATH).run()
    app.session_state["english_quest_session"] = new_english_quest_session(
        history=opening_history,
    )

    app.run()
    app.chat_input[0].set_value("测试附件提交").run()

    assert not app.exception
    parser.assert_called_once()
    invoke.assert_called_once_with(
        "V2",
        "请看图里的英语题",
        history=opening_history,
        attachment=submission.attachment,
    )
    assert app.session_state["english_quest_session"]["history"][-2:] == [
        {
            "role": "user",
            "content": "请看图里的英语题\n\n附件：question.png",
        },
        {
            "role": "assistant",
            "content": "关卡：2/5 生命：❤️❤️❤️ 经验：10 XP\n图片线索已识别。",
        },
    ]

    app.run()

    assert invoke.call_count == 1


def test_english_quest_attachment_parse_error_never_calls_agent(
    monkeypatch,
) -> None:
    opening_history = [
        {"role": "user", "content": "开始游戏"},
        {"role": "assistant", "content": "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP"},
    ]
    parser = Mock(side_effect=ChatSubmissionError("附件内容已经损坏。"))
    invoke = Mock()
    monkeypatch.setattr(chat_submission_module, "parse_chat_submission", parser)
    monkeypatch.setattr(facade_module, "invoke", invoke)
    app = AppTest.from_file(PAGE_PATH).run()
    app.session_state["english_quest_session"] = new_english_quest_session(
        history=opening_history,
    )

    app.run()
    app.chat_input[0].set_value("尝试上传附件").run()

    assert not app.exception
    parser.assert_called_once()
    invoke.assert_not_called()
    assert app.session_state["english_quest_session"]["history"] == opening_history
    assert "附件内容已经损坏" in app.session_state["english_quest_last_error"]


def test_english_quest_attachment_model_error_requires_reselection(
    monkeypatch,
) -> None:
    opening_history = [
        {"role": "user", "content": "开始游戏"},
        {"role": "assistant", "content": "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP"},
    ]
    submission = _image_submission()
    monkeypatch.setattr(
        chat_submission_module,
        "parse_chat_submission",
        Mock(return_value=submission),
    )
    invoke = Mock(return_value=new_agent_result("V2", error="模型暂时不可用。"))
    monkeypatch.setattr(facade_module, "invoke", invoke)
    app = AppTest.from_file(PAGE_PATH).run()
    app.session_state["english_quest_session"] = new_english_quest_session(
        history=opening_history,
    )

    app.run()
    app.chat_input[0].set_value("测试附件失败").run()

    assert not app.exception
    invoke.assert_called_once_with(
        "V2",
        "",
        history=opening_history,
        attachment=submission.attachment,
    )
    assert app.session_state["english_quest_session"]["history"] == opening_history
    assert "附件未保留，请重新选择附件后发送" in app.session_state[
        "english_quest_last_error"
    ]
