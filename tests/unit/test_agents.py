from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import src.agents as agents_module
from src.model import ModelConfigurationError
from src.schemas import ChatMessage


class FakeLlm:
    def __init__(self, content: object) -> None:
        self.content = content
        self.messages: list[str] = []

    def invoke(self, message: str) -> SimpleNamespace:
        self.messages.append(message)
        return SimpleNamespace(content=self.content)


class FakeAgent:
    def __init__(self, content: str, inputs: list[dict]) -> None:
        self.content = content
        self.inputs = inputs

    def invoke(self, state: dict) -> dict:
        self.inputs.append(state)
        return {"messages": [SimpleNamespace(content=self.content)]}


class ToolCallingFakeModel(BaseChatModel):
    responses: list[AIMessage]
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(messages)
        response = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_invoke_v0_returns_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLlm("  你好，我是学习助手。  ")
    monkeypatch.setattr(agents_module, "get_llm", lambda: fake_llm)

    result = agents_module.invoke_v0(" 你好 ")

    assert fake_llm.messages == ["你好"]
    assert result == {
        "text": "你好，我是学习助手。",
        "stage": "V0",
        "tool_calls": [],
        "citations": [],
        "trace": [],
        "waiting_for": None,
        "error": None,
    }


def test_invoke_v0_reads_text_content_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLlm([{"type": "text", "text": "第一段"}, "第二段"])
    monkeypatch.setattr(agents_module, "get_llm", lambda: fake_llm)

    result = agents_module.invoke_v0("测试")

    assert result["text"] == "第一段\n第二段"
    assert result["error"] is None


def test_invoke_v0_rejects_empty_message() -> None:
    result = agents_module.invoke_v0("   ")

    assert result["text"] == ""
    assert result["error"] == "消息不能为空，请输入一个问题后重试。"


def test_invoke_v0_returns_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_configuration_error() -> None:
        raise ModelConfigurationError("缺少测试配置。")

    monkeypatch.setattr(agents_module, "get_llm", raise_configuration_error)

    result = agents_module.invoke_v0("测试")

    assert result["error"] == "缺少测试配置。"


def test_invoke_v0_returns_actionable_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLlm("不会返回")

    def fail(_: str) -> SimpleNamespace:
        raise TimeoutError("secret details should not escape")

    fake_llm.invoke = fail
    monkeypatch.setattr(agents_module, "get_llm", lambda: fake_llm)

    result = agents_module.invoke_v0("测试")

    assert result["error"] is not None
    assert "检查网络、API Key、账户余额和模型名称" in result["error"]
    assert "secret details" not in result["error"]


def test_invoke_v1_reloads_prompt_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Prompt A", encoding="utf-8")
    fake_llm = object()
    monkeypatch.setattr(agents_module, "get_llm", lambda: fake_llm)
    prompts: list[str] = []
    inputs: list[dict] = []

    def fake_create_agent(*, model: object, tools: list, system_prompt: str) -> FakeAgent:
        assert model is fake_llm
        assert tools == []
        prompts.append(system_prompt)
        return FakeAgent("请先观察时间线索，好吗？", inputs)

    monkeypatch.setattr(agents_module, "create_agent", fake_create_agent)

    first_result = agents_module.invoke_v1("  测试题目  ", prompt_path)
    prompt_path.write_text("Prompt B", encoding="utf-8")
    second_result = agents_module.invoke_v1("测试题目", prompt_path)

    assert prompts == ["Prompt A", "Prompt B"]
    assert inputs == [
        {"messages": [{"role": "user", "content": "测试题目"}]},
        {"messages": [{"role": "user", "content": "测试题目"}]},
    ]
    assert first_result["stage"] == "V1"
    assert first_result["text"] == "请先观察时间线索，好吗？"
    assert second_result["error"] is None


def test_invoke_v1_returns_prompt_error_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    get_llm = Mock()
    monkeypatch.setattr(agents_module, "get_llm", get_llm)

    result = agents_module.invoke_v1("测试", tmp_path / "missing.md")

    assert result["stage"] == "V1"
    assert result["error"] is not None
    assert "missing.md" in result["error"]
    get_llm.assert_not_called()


def test_invoke_v1_forwards_complete_history_without_mutating_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("一次只问一个问题。", encoding="utf-8")
    monkeypatch.setattr(agents_module, "get_llm", lambda: object())
    inputs: list[dict] = []
    monkeypatch.setattr(
        agents_module,
        "create_agent",
        lambda **kwargs: FakeAgent("这个线索说明动作持续到什么时候？", inputs),
    )
    history: list[ChatMessage] = [
        {"role": "user", "content": "这道题怎么做？"},
        {"role": "assistant", "content": "你先找到了哪个时间线索？"},
    ]
    original_history = [message.copy() for message in history]

    result = agents_module.invoke_v1(
        "  我找到了 three times。  ",
        prompt_path,
        history=history,
    )

    assert inputs == [
        {
            "messages": [
                {"role": "user", "content": "这道题怎么做？"},
                {"role": "assistant", "content": "你先找到了哪个时间线索？"},
                {"role": "user", "content": "我找到了 three times。"},
            ]
        }
    ]
    assert history == original_history
    assert result["text"] == "这个线索说明动作持续到什么时候？"


def test_invoke_v1_rejects_incomplete_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_agent = Mock()
    monkeypatch.setattr(agents_module, "create_agent", create_agent)
    history: list[ChatMessage] = [
        {"role": "user", "content": "还没有对应的教练回复。"},
    ]

    result = agents_module.invoke_v1("继续", history=history)

    assert result["error"] is not None
    assert "缺少教练" in result["error"]
    create_agent.assert_not_called()


def test_invoke_v1_rejects_empty_message() -> None:
    result = agents_module.invoke_v1("   ")

    assert result["text"] == ""
    assert result["error"] == "消息不能为空，请输入一个问题后重试。"


def _write_test_skill(skills_path: Path, description: str, step: str) -> Path:
    skill_dir = skills_path / "sorting-out-mistakes"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        f"description: {description}\n"
        "---\n"
        "\n"
        f"Step 1：{step}\n",
        encoding="utf-8",
    )
    return skill_path


def test_invoke_v2_exposes_metadata_and_loads_full_skill_on_demand(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("一次只问一个问题。", encoding="utf-8")
    skills_path = tmp_path / "skill"
    _write_test_skill(skills_path, "用户要求整理错题时使用", "先收集原题。")
    fake_llm = object()
    monkeypatch.setattr(agents_module, "get_llm", lambda: fake_llm)
    loaded_instructions: list[str] = []
    inputs: list[dict] = []

    def fake_create_agent(*, model: object, tools: list, system_prompt: str):
        assert model is fake_llm
        assert len(tools) == 3
        assert tools[0].name == "load_skill"
        assert tools[1].name == "load_mistake_file"
        assert tools[2].name == "save_mistake"
        assert "sorting-out-mistakes: 用户要求整理错题时使用" in tools[0].description
        assert "sorting-out-mistakes: 用户要求整理错题时使用" in system_prompt
        assert "先收集原题" not in system_prompt
        assert "Skill 加载后，其任务步骤优先于" in system_prompt
        assert "先调用 load_mistake_file" in system_prompt
        assert "立即调用 save_mistake" in system_prompt

        class FakeV2Agent:
            def invoke(self, state: dict) -> dict:
                inputs.append(state)
                loaded_instructions.append(
                    tools[0].invoke({"skill_name": "sorting-out-mistakes"})
                )
                return {
                    "messages": [
                        SimpleNamespace(
                            content="",
                            tool_calls=[
                                {
                                    "name": "load_skill",
                                    "args": {
                                        "skill_name": "sorting-out-mistakes",
                                    },
                                    "id": "call-1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        SimpleNamespace(
                            content="请先把错题原文发给我，可以吗？",
                            tool_calls=[],
                        ),
                    ]
                }

        return FakeV2Agent()

    monkeypatch.setattr(agents_module, "create_agent", fake_create_agent)

    result = agents_module.invoke_v2(
        " 请帮我整理错题 ",
        prompt_path,
        skills_path,
    )

    assert loaded_instructions == ["Step 1：先收集原题。"]
    assert inputs == [
        {"messages": [{"role": "user", "content": "请帮我整理错题"}]}
    ]
    assert result["text"] == "请先把错题原文发给我，可以吗？"
    assert result["tool_calls"] == [
        {
            "name": "load_skill",
            "args": {"skill_name": "sorting-out-mistakes"},
            "id": "call-1",
            "type": "tool_call",
        }
    ]


def test_invoke_v2_reloads_skill_instructions_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("基础 Prompt", encoding="utf-8")
    skills_path = tmp_path / "skill"
    skill_path = _write_test_skill(skills_path, "整理错题时使用", "第一版线索。")
    monkeypatch.setattr(agents_module, "get_llm", lambda: object())
    loaded_instructions: list[str] = []

    def fake_create_agent(*, model: object, tools: list, system_prompt: str):
        class FakeV2Agent:
            def invoke(self, state: dict) -> dict:
                loaded_instructions.append(
                    tools[0].invoke({"skill_name": "sorting-out-mistakes"})
                )
                return {
                    "messages": [SimpleNamespace(content="继续整理。", tool_calls=[])]
                }

        return FakeV2Agent()

    monkeypatch.setattr(agents_module, "create_agent", fake_create_agent)

    first = agents_module.invoke_v2("整理错题", prompt_path, skills_path)
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        "description: 整理错题时使用\n"
        "---\n"
        "\n"
        "Step 1：第二版线索。\n",
        encoding="utf-8",
    )
    second = agents_module.invoke_v2("继续整理", prompt_path, skills_path)

    assert first["error"] is None
    assert second["error"] is None
    assert loaded_instructions == [
        "Step 1：第一版线索。",
        "Step 1：第二版线索。",
    ]


def test_invoke_v2_executes_load_skill_through_langchain_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_name": "sorting-out-mistakes"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="请先发来错题原文，可以吗？"),
        ]
    )
    monkeypatch.setattr(agents_module, "get_llm", lambda: model)

    result = agents_module.invoke_v2("请帮我整理错题")

    assert result["error"] is None
    assert result["text"] == "请先发来错题原文，可以吗？"
    assert model.bound_tool_names == [
        "load_skill",
        "load_mistake_file",
        "save_mistake",
    ]
    assert [type(message).__name__ for message in model.seen_messages[-1]][-2:] == [
        "AIMessage",
        "ToolMessage",
    ]
    assert "Step 1" in model.seen_messages[-1][-1].content
    assert result["tool_calls"][0]["name"] == "load_skill"


def test_load_mistake_file_tool_reads_only_markdown_inside_inbox(
    tmp_path: Path,
) -> None:
    mistakes_root = tmp_path / "student" / "mistakes"
    inbox_path = mistakes_root / "inbox"
    records_path = mistakes_root / "records" / "english"
    inbox_path.mkdir(parents=True)
    records_path.mkdir(parents=True)
    source_path = inbox_path / "english.md"
    source_path.write_text(
        "错题1\n原题：第一题\n\n错题2\n原题：第二题\n",
        encoding="utf-8",
    )
    record_path = records_path / "mistake-existing.md"
    record_path.write_text("不应作为批量输入读取", encoding="utf-8")
    load_mistake_file = agents_module._create_load_mistake_file_tool(inbox_path)

    success = load_mistake_file.invoke({"path": str(source_path)})
    failure = load_mistake_file.invoke({"path": str(record_path)})

    assert "读取成功" in success
    assert "错题1" in success
    assert "错题2" in success
    assert failure.startswith("读取失败：")


def test_save_mistake_tool_writes_once_and_returns_relative_path(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "student" / "mistakes" / "records"
    inbox_path = tmp_path / "student" / "mistakes" / "inbox"
    save_mistake = agents_module._create_save_mistake_tool(
        records_path,
        records_path,
        inbox_path,
    )
    mistake = {
        "subject": "英语",
        "topic": "Present Perfect",
        "source": "chat",
        "problem_type": "语法填空",
        "original_question": "I ____ (read) this book three times.",
        "student_answer": "am reading",
        "correct_answer": "have read",
        "correct_reasoning": "three times 表示截至现在已经发生三次。",
        "error_reason": "混淆了现在进行时和现在完成时。",
        "knowledge_point": "现在完成时：have/has + 过去分词。",
        "next_reminder": "先圈出次数和完成标志词。",
    }

    first_result = save_mistake.invoke(mistake)
    second_result = save_mistake.invoke(mistake)

    saved_files = list((records_path / "english").glob("mistake-*.md"))
    assert len(saved_files) == 1
    assert saved_files[0].parent.name == "english"
    assert "保存成功" in first_result
    assert saved_files[0].name in first_result
    assert "已经保存" in second_result
    assert saved_files[0].name in second_result
    content = saved_files[0].read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "schema_version: 1" in content
    assert f"id: {saved_files[0].stem}" in content
    assert "subject: english" in content
    assert "topic: present-perfect" in content
    assert "status: needs-review" in content
    assert f'created_at: "{date.today().isoformat()}"' in content
    assert "review_count: 0" in content
    assert "next_review_at: null" in content
    assert 'source: "chat"' in content
    assert "- 学科：英语" in content
    assert "- 题型：语法填空" in content
    assert "- 正确答案：have read" in content


def test_save_mistake_tool_rejects_source_outside_inbox(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "student" / "mistakes" / "records"
    inbox_path = tmp_path / "student" / "mistakes" / "inbox"
    save_mistake = agents_module._create_save_mistake_tool(
        records_path,
        records_path,
        inbox_path,
    )

    result = save_mistake.invoke(
        {
            "subject": "英语",
            "topic": "present-perfect",
            "source": str(tmp_path / "private.md"),
            "problem_type": "语法填空",
            "original_question": "测试题",
            "student_answer": "错误答案",
            "correct_answer": "待补充",
            "correct_reasoning": "待补充",
            "error_reason": "待补充",
            "knowledge_point": "待补充",
            "next_reminder": "待补充",
        }
    )

    assert result.startswith("保存失败：")
    assert "source 必须是 chat 或 inbox/ 内的 Markdown 路径" in result
    assert not records_path.exists()


def test_save_mistake_tool_rejects_non_slug_topic(tmp_path: Path) -> None:
    records_path = tmp_path / "student" / "mistakes" / "records"
    inbox_path = tmp_path / "student" / "mistakes" / "inbox"
    save_mistake = agents_module._create_save_mistake_tool(
        records_path,
        records_path,
        inbox_path,
    )

    result = save_mistake.invoke(
        {
            "subject": "英语",
            "topic": "现在完成时",
            "source": "chat",
            "problem_type": "语法填空",
            "original_question": "测试题",
            "student_answer": "错误答案",
            "correct_answer": "待补充",
            "correct_reasoning": "待补充",
            "error_reason": "待补充",
            "knowledge_point": "现在完成时",
            "next_reminder": "待补充",
        }
    )

    assert result.startswith("保存失败：")
    assert "topic 必须使用英文 kebab-case" in result
    assert not records_path.exists()


def test_save_mistake_tool_reports_failure_outside_records_root(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "student" / "mistakes" / "records"
    inbox_path = tmp_path / "student" / "mistakes" / "inbox"
    outside_path = tmp_path / "outside"
    save_mistake = agents_module._create_save_mistake_tool(
        outside_path,
        records_path,
        inbox_path,
    )

    result = save_mistake.invoke(
        {
            "subject": "英语",
            "topic": "present-perfect",
            "source": "chat",
            "problem_type": "语法填空",
            "original_question": "I ____ (read) this book three times.",
            "student_answer": "am reading",
            "correct_answer": "待补充",
            "correct_reasoning": "待补充",
            "error_reason": "待补充",
            "knowledge_point": "待补充",
            "next_reminder": "待补充",
        }
    )

    assert result.startswith("保存失败：")
    assert not outside_path.exists()


@pytest.mark.parametrize(
    ("subject", "directory"),
    [("英语", "english"), ("English", "english"), ("数学", "math")],
)
def test_save_mistake_tool_groups_known_subjects(
    tmp_path: Path,
    subject: str,
    directory: str,
) -> None:
    records_path = tmp_path / "student" / "mistakes" / "records"
    inbox_path = tmp_path / "student" / "mistakes" / "inbox"
    save_mistake = agents_module._create_save_mistake_tool(
        records_path,
        records_path,
        inbox_path,
    )

    result = save_mistake.invoke(
        {
            "subject": subject,
            "topic": "general",
            "source": "chat",
            "problem_type": "测试",
            "original_question": f"{subject}题目",
            "student_answer": "错误答案",
            "correct_answer": "待补充",
            "correct_reasoning": "待补充",
            "error_reason": "待补充",
            "knowledge_point": "待补充",
            "next_reminder": "待补充",
        }
    )

    assert "保存成功" in result
    assert len(list((records_path / directory).glob("mistake-*.md"))) == 1


def test_invoke_v2_executes_load_then_save_through_langchain_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    student_root = tmp_path / "student"
    inbox_path = student_root / "mistakes" / "inbox"
    records_path = student_root / "mistakes" / "records"
    mistake = {
        "subject": "英语",
        "topic": "present-perfect",
        "source": "chat",
        "problem_type": "语法填空",
        "original_question": "I ____ (read) this book three times.",
        "student_answer": "am reading",
        "correct_answer": "have read",
        "correct_reasoning": "three times 表示截至现在已经发生三次。",
        "error_reason": "混淆了现在进行时和现在完成时。",
        "knowledge_point": "现在完成时：have/has + 过去分词。",
        "next_reminder": "先圈出次数和完成标志词。",
    }
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_name": "sorting-out-mistakes"},
                        "id": "call-load",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_mistake",
                        "args": mistake,
                        "id": "call-save",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="错题已经保存。"),
        ]
    )
    monkeypatch.setattr(agents_module, "get_llm", lambda: model)
    monkeypatch.setattr(agents_module, "MISTAKES_INBOX_PATH", inbox_path)
    monkeypatch.setattr(agents_module, "MISTAKES_RECORDS_PATH", records_path)

    result = agents_module.invoke_v2(
        "请整理这道错题：I ____ (read) this book three times. 我的答案是 am reading。"
    )

    assert result["error"] is None
    assert result["text"] == "错题已经保存。"
    assert [call["name"] for call in result["tool_calls"]] == [
        "load_skill",
        "save_mistake",
    ]
    assert len(list((records_path / "english").glob("mistake-*.md"))) == 1


def test_invoke_v2_reads_two_mistakes_and_saves_each_through_langchain_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    student_root = tmp_path / "student"
    inbox_path = student_root / "mistakes" / "inbox"
    records_path = student_root / "mistakes" / "records"
    inbox_path.mkdir(parents=True)
    source_path = inbox_path / "english.md"
    source_path.write_text(
        "错题1\n"
        "类型：语法填空\n"
        "原题：I ____ (read) this book three times.\n"
        "我的答案：am reading\n\n"
        "错题2\n"
        "类型：语法填空\n"
        "原题：She ____ (go) to the library yesterday.\n"
        "我的答案：has gone\n",
        encoding="utf-8",
    )
    first_mistake = {
        "subject": "英语",
        "topic": "present-perfect",
        "source": str(source_path),
        "problem_type": "语法填空",
        "original_question": "I ____ (read) this book three times.",
        "student_answer": "am reading",
        "correct_answer": "have read",
        "correct_reasoning": "待补充",
        "error_reason": "待补充",
        "knowledge_point": "现在完成时",
        "next_reminder": "待补充",
    }
    second_mistake = {
        "subject": "英语",
        "topic": "simple-past",
        "source": str(source_path),
        "problem_type": "语法填空",
        "original_question": "She ____ (go) to the library yesterday.",
        "student_answer": "has gone",
        "correct_answer": "went",
        "correct_reasoning": "待补充",
        "error_reason": "待补充",
        "knowledge_point": "一般过去时",
        "next_reminder": "待补充",
    }
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_name": "sorting-out-mistakes"},
                        "id": "call-load-skill",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_mistake_file",
                        "args": {"path": str(source_path)},
                        "id": "call-load-file",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_mistake",
                        "args": first_mistake,
                        "id": "call-save-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "save_mistake",
                        "args": second_mistake,
                        "id": "call-save-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="已读取 2 道错题并分别保存。"),
        ]
    )
    monkeypatch.setattr(agents_module, "get_llm", lambda: model)
    monkeypatch.setattr(agents_module, "MISTAKES_INBOX_PATH", inbox_path)
    monkeypatch.setattr(agents_module, "MISTAKES_RECORDS_PATH", records_path)

    result = agents_module.invoke_v2(f"继续整理 {source_path}")

    assert result["error"] is None
    assert result["text"] == "已读取 2 道错题并分别保存。"
    assert [call["name"] for call in result["tool_calls"]] == [
        "load_skill",
        "load_mistake_file",
        "save_mistake",
        "save_mistake",
    ]
    assert len(list((records_path / "english").glob("mistake-*.md"))) == 2
    saved_contents = [
        path.read_text(encoding="utf-8")
        for path in (records_path / "english").glob("mistake-*.md")
    ]
    assert all('source: "inbox/english.md"' in content for content in saved_contents)
    assert any("topic: present-perfect" in content for content in saved_contents)
    assert any("topic: simple-past" in content for content in saved_contents)


def test_invoke_v2_returns_skill_error_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("基础 Prompt", encoding="utf-8")
    get_llm = Mock()
    monkeypatch.setattr(agents_module, "get_llm", get_llm)

    result = agents_module.invoke_v2(
        "整理错题",
        prompt_path,
        tmp_path / "missing-skills",
    )

    assert result["stage"] == "V2"
    assert result["error"] is not None
    assert "missing-skills" in result["error"]
    get_llm.assert_not_called()


def test_invoke_v2_rejects_empty_message() -> None:
    result = agents_module.invoke_v2("   ")

    assert result["text"] == ""
    assert result["error"] == "消息不能为空，请输入一个问题后重试。"
