"""V0 到 V3 的 Agent 调用逻辑。"""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.tools import tool

from src.artifacts import (
    ArtifactError,
    PROJECT_ROOT,
    PROMPT_PATH,
    SKILLS_PATH,
    SkillMetadata,
    discover_skills,
    read_markdown,
    read_skill,
)
from src.model import ModelConfigurationError, get_llm
from src.schemas import AgentResult, ChatMessage, new_agent_result
from src.storage import (
    MISTAKES_PATH,
    STUDENT_ROOT,
    FileAlreadyExistsError,
    StorageError,
    save_markdown,
)


class ConversationHistoryError(ValueError):
    """Agent 对话历史不是完整的学生、教练消息对。"""


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


def _tool_calls_from_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            calls = message.get("tool_calls", [])
        else:
            calls = getattr(message, "tool_calls", [])
        if not isinstance(calls, list):
            continue
        tool_calls.extend(dict(call) for call in calls if isinstance(call, dict))
    return tool_calls


def _create_load_skill_tool(skills: Sequence[SkillMetadata]):
    skill_paths = {skill.name: skill.path for skill in skills}
    available_names = "、".join(skill_paths)
    catalog = "\n".join(
        f"- {skill.name}: {skill.description}" for skill in skills
    )

    @tool(
        description=(
            "按名称加载专业 Skill 的完整操作说明。\n\n"
            f"可用 Skills：\n{catalog}\n\n"
            "返回该 Skill 的提示词和操作步骤。"
        )
    )
    def load_skill(skill_name: str) -> str:
        normalized_name = skill_name.strip()
        skill_path = skill_paths.get(normalized_name)
        if skill_path is None:
            return (
                f"未找到 Skill：{normalized_name}。"
                f"当前可用 Skill：{available_names}。"
            )
        return read_skill(skill_path).instructions

    return load_skill


def _mistake_markdown(
    *,
    subject: str,
    problem_type: str,
    original_question: str,
    student_answer: str,
    correct_answer: str,
    correct_reasoning: str,
    error_reason: str,
    knowledge_point: str,
    next_reminder: str,
) -> str:
    fields = {
        "学科": subject,
        "题型": problem_type,
        "原题": original_question,
        "我的答案": student_answer,
        "正确答案": correct_answer,
        "正确思路": correct_reasoning,
        "错因": error_reason,
        "知识点": knowledge_point,
        "下次提醒": next_reminder,
    }
    lines = ["# 错题记录", ""]
    lines.extend(
        f"- {label}：{' '.join(value.split()) or '待补充'}"
        for label, value in fields.items()
    )
    return "\n".join(lines)


def _display_output_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _create_save_mistake_tool(
    mistakes_path: str | Path,
    allowed_root: str | Path,
):
    @tool(
        description=(
            "把已经整理好的单道错题保存为 student/mistakes/ 下的 Markdown。"
            "仅在 load_skill 已加载错题整理 Skill，且原题和学生原答案已知时调用。"
            "暂时未知的分析字段传入‘待补充’，不要编造。"
        )
    )
    def save_mistake(
        subject: Annotated[str, "学科，例如英语或数学"],
        problem_type: Annotated[str, "题型，例如语法填空"],
        original_question: Annotated[str, "完整原题"],
        student_answer: Annotated[str, "学生当时的错误答案"],
        correct_answer: Annotated[str, "正确答案，未知时写待补充"],
        correct_reasoning: Annotated[str, "正确思路，未知时写待补充"],
        error_reason: Annotated[str, "错因，未知时写待补充"],
        knowledge_point: Annotated[str, "对应知识点，未知时写待补充"],
        next_reminder: Annotated[str, "下次可执行的检查方法，未知时写待补充"],
    ) -> str:
        """保存一条结构化错题记录。"""

        clean_question = " ".join(original_question.split())
        clean_student_answer = " ".join(student_answer.split())
        if not clean_question or not clean_student_answer:
            return "保存失败：原题和学生原答案不能为空。"

        identity = "\n".join(
            [
                " ".join(subject.split()).casefold(),
                " ".join(problem_type.split()).casefold(),
                clean_question.casefold(),
                clean_student_answer.casefold(),
            ]
        )
        digest = sha256(identity.encode("utf-8")).hexdigest()[:12]
        filename = f"mistake-{digest}.md"
        markdown = _mistake_markdown(
            subject=subject,
            problem_type=problem_type,
            original_question=original_question,
            student_answer=student_answer,
            correct_answer=correct_answer,
            correct_reasoning=correct_reasoning,
            error_reason=error_reason,
            knowledge_point=knowledge_point,
            next_reminder=next_reminder,
        )

        try:
            saved_path = save_markdown(
                mistakes_path,
                filename,
                markdown,
                allowed_root=allowed_root,
            )
        except FileAlreadyExistsError:
            existing_path = Path(mistakes_path) / filename
            return (
                "这道错题已经保存，无需重复写入："
                f"{_display_output_path(existing_path)}"
            )
        except StorageError as exc:
            return f"保存失败：{exc}"

        return f"保存成功：{_display_output_path(saved_path)}"

    return save_mistake


def _v2_system_prompt(
    base_prompt: str,
    skills: Sequence[SkillMetadata],
) -> str:
    catalog = "\n".join(
        f"- {skill.name}: {skill.description}" for skill in skills
    )
    return (
        f"{base_prompt}\n\n"
        "# 可按需加载的 Skills\n"
        f"{catalog}\n\n"
        "只有当当前请求或对话中的持续任务符合某项 description 时，"
        "才调用 load_skill。"
        "调用时必须传入目录中列出的准确 Skill 名称。"
        "工具返回完整操作说明后，按照其中的步骤继续当前任务。"
        "不符合任何 Skill 时，不要调用 load_skill。\n\n"
        "# Skill 执行优先级\n"
        "当当前消息直接包含错题、题型、原题、我的答案等结构化错题信息时，"
        "也应视为错题整理需求并加载匹配 Skill。"
        "Skill 加载后，其任务步骤优先于上面的基础教学 Prompt。"
        "只在 Skill 要求补充信息时使用一次一个问题的苏格拉底方式。"
        "如果 Skill 允许把未知字段记为待补充，就不要为了凑齐所有字段反复追问。"
        "当 Skill 要求写入且保存条件已经满足时，立即调用 save_mistake。"
        "只有 save_mistake 返回保存成功或已经保存后，才能告诉学生文件已保存；"
        "如果工具返回保存失败，必须如实说明失败原因。"
    )


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


def invoke_v2(
    message: str,
    prompt_path: str | Path = PROMPT_PATH,
    skills_path: str | Path = SKILLS_PATH,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> AgentResult:
    """按需加载匹配的标准 SKILL.md，完成一轮 V2 对话。"""

    clean_message = message.strip()
    if not clean_message:
        return new_agent_result("V2", error="消息不能为空，请输入一个问题后重试。")

    try:
        conversation = _validated_history(history)
        base_prompt = read_markdown(prompt_path)
        skills = discover_skills(skills_path)
        agent = create_agent(
            model=get_llm(),
            tools=[
                _create_load_skill_tool(skills),
                _create_save_mistake_tool(MISTAKES_PATH, STUDENT_ROOT),
            ],
            system_prompt=_v2_system_prompt(base_prompt, skills),
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
        return new_agent_result("V2", error=str(exc))
    except Exception as exc:
        return new_agent_result(
            "V2",
            error=(
                "V2 调用失败。请检查 Prompt、Skill、网络、API Key、账户余额和模型名称后重试。"
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
        return new_agent_result("V2", error="V2 返回了空内容，请稍后重试。")
    return new_agent_result(
        "V2",
        text=text,
        tool_calls=_tool_calls_from_messages(messages),
    )
