import pytest

import src.facade as facade_module
from src.schemas import ChatMessage, new_agent_result


def test_invoke_rejects_unavailable_stage() -> None:
    result = facade_module.invoke("V4", "测试")

    assert result["stage"] == "V4"
    assert result["error"] == (
        "当前版本仅支持 V0、V1、V2 和 V3，收到的阶段为 V4。"
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
