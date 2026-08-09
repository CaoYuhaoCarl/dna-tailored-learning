"""V0 到 V3 的 Agent 调用逻辑。"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain.agents import create_agent

from src.artifacts import ArtifactError, PROMPT_PATH, read_markdown
from src.model import ModelConfigurationError, get_llm
from src.schemas import AgentResult, ChatMessage, new_agent_result


class ConversationHistoryError(ValueError):
    """V1 对话历史不是完整的学生、教练消息对。"""


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


def _validated_history(
    history: Sequence[ChatMessage] | None,
) -> list[ChatMessage]:
    if history is None:
        return []

    normalized: list[ChatMessage] = []
    for index, message in enumerate(history):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected_role:
            raise ConversationHistoryError(
                f"对话历史第 {index + 1} 条消息应为 {expected_role}。"
                "请按学生、教练的顺序成对保存消息。"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ConversationHistoryError(
                f"对话历史第 {index + 1} 条消息内容为空。请删除或补全该消息。"
            )
        normalized.append({"role": expected_role, "content": content.strip()})

    if len(normalized) % 2 != 0:
        raise ConversationHistoryError(
            "对话历史缺少教练对上一条学生消息的回复。请保存完整的一问一答后重试。"
        )
    return normalized


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

    text = _response_text(getattr(response, "content", None))
    if not text:
        return new_agent_result(
            "V0",
            error="DeepSeek 返回了空内容，请稍后重试。",
        )
    return new_agent_result("V0", text=text)


def invoke_v1(
    message: str,
    prompt_path: str | Path = PROMPT_PATH,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> AgentResult:
    """实时读取学生 Prompt 和对话历史，完成一轮苏格拉底问答。"""

    clean_message = message.strip()
    if not clean_message:
        return new_agent_result("V1", error="消息不能为空，请输入一个问题后重试。")

    try:
        conversation = _validated_history(history)
        system_prompt = read_markdown(prompt_path)
        agent = create_agent(
            model=get_llm(),
            tools=[],
            system_prompt=system_prompt,
        )
        state = agent.invoke(
            {
                "messages": [
                    *conversation,
                    {"role": "user", "content": clean_message},
                ]
            }
        )
    except (ArtifactError, ConversationHistoryError, ModelConfigurationError) as exc:
        return new_agent_result("V1", error=str(exc))
    except Exception as exc:
        return new_agent_result(
            "V1",
            error=(
                "V1 调用失败。请检查 Prompt、网络、API Key、账户余额和模型名称后重试。"
                f"错误类型：{type(exc).__name__}。"
            ),
        )

    messages = state.get("messages", [])
    final_message = messages[-1] if messages else None
    if isinstance(final_message, dict):
        content = final_message.get("content")
    else:
        content = getattr(final_message, "content", None)
    text = _response_text(content)
    if not text:
        return new_agent_result("V1", error="V1 返回了空内容，请稍后重试。")
    return new_agent_result("V1", text=text)
