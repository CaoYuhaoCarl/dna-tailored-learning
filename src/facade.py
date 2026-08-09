"""Notebook 和后续 Streamlit 共用的稳定入口。"""

from collections.abc import Sequence

from src.agents import invoke_v0, invoke_v1, invoke_v2, invoke_v3
from src.schemas import AgentResult, ChatMessage, new_agent_result


def invoke(
    stage: str,
    message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> AgentResult:
    """调用 V0 到 V3，V1-V3 可接收由界面保存的完整对话历史。"""

    normalized_stage = stage.strip().upper()
    if normalized_stage == "V0":
        return invoke_v0(message)
    if normalized_stage == "V1":
        return invoke_v1(message, history=history)
    if normalized_stage == "V2":
        return invoke_v2(message, history=history)
    if normalized_stage == "V3":
        return invoke_v3(message, history=history)

    display_stage = normalized_stage or "UNKNOWN"
    return new_agent_result(
        display_stage,
        error=(
            "当前版本仅支持 V0、V1、V2 和 V3，"
            f"收到的阶段为 {display_stage}。"
        ),
    )
