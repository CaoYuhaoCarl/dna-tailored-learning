"""Notebook 和后续 Streamlit 共用的稳定入口。"""

from src.agents import invoke_v0
from src.schemas import AgentResult, new_agent_result


def invoke(stage: str, message: str) -> AgentResult:
    """调用指定阶段，第一阶段仅开放 V0。"""

    normalized_stage = stage.strip().upper()
    if normalized_stage == "V0":
        return invoke_v0(message)

    display_stage = normalized_stage or "UNKNOWN"
    return new_agent_result(
        display_stage,
        error=f"当前版本仅支持 V0，收到的阶段为 {display_stage}。",
    )
