"""V0 到 V3 的 Agent 调用逻辑。"""

from typing import Any

from src.model import ModelConfigurationError, get_llm
from src.schemas import AgentResult, new_agent_result


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def invoke_v0(message: str) -> AgentResult:
    """直接调用 DeepSeek，不挂载 Prompt、Skill、Knowledge 或 Workflow。"""

    clean_message = message.strip()
    if not clean_message:
        return new_agent_result("V0", error="消息不能为空，请输入一个问题后重试。")

    try:
        response = get_llm().invoke(clean_message)
    except ModelConfigurationError as exc:
        return new_agent_result("V0", error=str(exc))
    except Exception as exc:
        return new_agent_result(
            "V0",
            error=(
                "DeepSeek 调用失败。请检查网络、API Key、账户余额和模型名称后重试。"
                f"错误类型：{type(exc).__name__}。"
            ),
        )

    text = _response_text(response.content)
    if not text:
        return new_agent_result(
            "V0",
            error="DeepSeek 返回了空内容，请稍后重试。",
        )
    return new_agent_result("V0", text=text)
