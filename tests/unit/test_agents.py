from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
