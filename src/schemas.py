"""跨版本共享的数据契约。"""

from typing import Any, TypedDict


class AgentResult(TypedDict):
    """V0 到 V4 对上层暴露的统一结果结构。"""

    text: str
    stage: str
    tool_calls: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    waiting_for: str | None
    error: str | None


def new_agent_result(
    stage: str,
    *,
    text: str = "",
    error: str | None = None,
) -> AgentResult:
    """创建字段完整的 Agent 结果。"""

    return {
        "text": text,
        "stage": stage,
        "tool_calls": [],
        "citations": [],
        "trace": [],
        "waiting_for": None,
        "error": error,
    }
