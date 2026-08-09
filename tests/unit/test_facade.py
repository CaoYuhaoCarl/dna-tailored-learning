import pytest

import src.facade as facade_module
from src.schemas import ChatMessage, new_agent_result


def test_invoke_rejects_unavailable_stage() -> None:
    result = facade_module.invoke("V2", "测试")

    assert result["stage"] == "V2"
    assert result["error"] == "当前版本仅支持 V0 和 V1，收到的阶段为 V2。"


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
