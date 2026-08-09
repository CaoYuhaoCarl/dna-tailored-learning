"""Notebook 和后续 Streamlit 共用的稳定入口。"""

from collections.abc import Sequence

from src.agents import invoke_v0, invoke_v1
from src.schemas import AgentResult, ChatMessage, new_agent_result


def invoke(
    stage: str,
    message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> AgentResult:
    """调用 V0 或 V1，V1 可接收由界面保存的完整对话历史。"""

    normalized_stage = stage.strip().upper()
    if normalized_stage == "V0":
        return invoke_v0(message)
    if normalized_stage == "V1":
        return invoke_v1(message, history=history)

    display_stage = normalized_stage or "UNKNOWN"
    return new_agent_result(
        display_stage,
        error=f"当前版本仅支持 V0 和 V1，收到的阶段为 {display_stage}。",
    )
