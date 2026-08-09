"""跨版本共享的数据契约。"""

from typing import Any, Literal, TypedDict


class ChatMessage(TypedDict):
    """V1 多轮对话中由调用方保存的一条消息。"""

    role: Literal["user", "assistant"]
    content: str


CitationField = Literal[
    "标题",
    "关键词",
    "别名",
    "学科",
    "年级",
    "核心规则",
    "例句",
    "易错提醒",
]


class CitationMatch(TypedDict):
    """知识卡中实际触发本轮检索的字段。"""

    field: CitationField
    terms: list[str]
    excerpt: str
    method: Literal["exact", "semantic"]


class Citation(TypedDict):
    """一条可追溯到学生知识卡原文的引用。"""

    id: str
    source: str
    title: str
    matches: list[CitationMatch]


class AgentResult(TypedDict):
    """V0 到 V4 对上层暴露的统一结果结构。"""

    text: str
    stage: str
    tool_calls: list[dict[str, Any]]
    citations: list[Citation]
    trace: list[dict[str, Any]]
    waiting_for: str | None
    error: str | None


def new_agent_result(
    stage: str,
    *,
    text: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    citations: list[Citation] | None = None,
    trace: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> AgentResult:
    """创建字段完整的 Agent 结果。"""

    return {
        "text": text,
        "stage": stage,
        "tool_calls": list(tool_calls or []),
        "citations": list(citations or []),
        "trace": list(trace or []),
        "waiting_for": None,
        "error": error,
    }
