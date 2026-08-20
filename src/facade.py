"""Notebook 和 Streamlit 共用的稳定入口。"""

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import unified_diff
from hashlib import sha256
import os
from pathlib import Path
import platform
import tempfile
from typing import TypedDict

from src.agents import invoke_v0, invoke_v1, invoke_v2, invoke_v3
from src.artifacts import (
    PROMPT_PATH,
    PROJECT_ROOT,
    SKILLS_PATH,
    ArtifactError,
    read_markdown,
    read_skill,
)
from src.chat_submission import ChatAttachment
from src.model import (
    ModelConfigurationError,
    ModelConfigurationSummary,
    get_model_configuration_summary as _get_model_configuration_summary,
    save_model_configuration as _save_model_configuration,
    validate_model_configuration,
)
from src.schemas import AgentResult, ChatMessage, new_agent_result
from src.workflow import chat_v4 as _chat_v4


class AppStatus(TypedDict):
    """首页展示所需的只读运行状态。"""

    ready: bool
    runtime_ready: bool
    model_ready: bool
    python_version: str
    expected_python_version: str | None
    model_provider: str | None
    missing_files: list[str]
    runtime_errors: list[str]
    model_error: str | None
    errors: list[str]


class LessonArtifact(TypedDict):
    """第一课页面可展示的单个 Markdown 文件快照。"""

    stage: str
    label: str
    path: str
    content: str
    digest: str


class LessonArtifactChange(TypedDict):
    """一次明确保存产生的前后快照和统一差异。"""

    before: LessonArtifact
    after: LessonArtifact
    changed: bool
    diff: str


LessonArtifactError = ArtifactError


class ArtifactConflictError(LessonArtifactError):
    """编辑期间磁盘文件已被其他页面修改。"""


@dataclass(frozen=True)
class _LessonArtifactSpec:
    stage: str
    label: str
    path: Path
    kind: str


_REQUIRED_APP_FILES = (
    ".python-version",
    "student/prompt.md",
    "student/skill/sorting-out-mistakes/SKILL.md",
    "student/v4-prompt.md",
    "student/knowledge/english/grammar/present-perfect.md",
)

_MAX_LESSON_ARTIFACT_BYTES = 256 * 1024
_LESSON_ARTIFACTS = {
    "V1": _LessonArtifactSpec(
        stage="V1",
        label="苏格拉底教练 Prompt",
        path=PROMPT_PATH,
        kind="prompt",
    ),
    "V2": _LessonArtifactSpec(
        stage="V2",
        label="整理错题 Skill",
        path=SKILLS_PATH / "sorting-out-mistakes" / "SKILL.md",
        kind="skill",
    ),
}


def _lesson_artifact_spec(stage: str) -> _LessonArtifactSpec | None:
    normalized_stage = stage.strip().upper()
    if normalized_stage == "V0":
        return None
    spec = _LESSON_ARTIFACTS.get(normalized_stage)
    if spec is None:
        display_stage = normalized_stage or "UNKNOWN"
        raise ArtifactError(
            "第一课只支持 V0、V1 和 V2，"
            f"收到的阶段为 {display_stage}。"
        )
    return spec


def _artifact_display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _validate_artifact_target(path: Path) -> None:
    student_root = (PROJECT_ROOT / "student").resolve()
    if path.is_symlink():
        raise ArtifactError(
            f"{_artifact_display_path(path)} 不能是符号链接。"
            "请从课程包恢复普通 Markdown 文件后重试。"
        )
    try:
        path.resolve().relative_to(student_root)
    except ValueError as exc:
        raise ArtifactError("第一课只能读取和保存 student/ 内的课程文件。") from exc


def _read_lesson_artifact(spec: _LessonArtifactSpec) -> LessonArtifact:
    _validate_artifact_target(spec.path)
    try:
        size = spec.path.stat().st_size
    except FileNotFoundError:
        size = 0
    except OSError as exc:
        raise ArtifactError(
            f"无法检查 {_artifact_display_path(spec.path)}。"
            "请确认文件可访问后重试。"
        ) from exc
    if size > _MAX_LESSON_ARTIFACT_BYTES:
        raise ArtifactError(
            f"{_artifact_display_path(spec.path)} 超过 256 KB。"
            "请缩短课程 Markdown 后重试。"
        )

    content = read_markdown(spec.path)
    if spec.kind == "skill":
        read_skill(spec.path)
    return {
        "stage": spec.stage,
        "label": spec.label,
        "path": _artifact_display_path(spec.path),
        "content": content,
        "digest": sha256(content.encode("utf-8")).hexdigest(),
    }


def get_lesson_artifact(stage: str) -> LessonArtifact | None:
    """读取 V1 Prompt 或 V2 Skill；V0 没有可编辑文件。"""

    spec = _lesson_artifact_spec(stage)
    if spec is None:
        return None
    return _read_lesson_artifact(spec)


def _artifact_diff(before: LessonArtifact, after: LessonArtifact) -> str:
    return "\n".join(
        unified_diff(
            before["content"].splitlines(),
            after["content"].splitlines(),
            fromfile=f"{before['path']}（修改前）",
            tofile=f"{after['path']}（修改后）",
            lineterm="",
        )
    )


def save_lesson_artifact(
    stage: str,
    content: str,
    *,
    expected_digest: str,
) -> LessonArtifactChange:
    """校验并原子保存 V1/V2 文件，同时防止覆盖外部修改。"""

    spec = _lesson_artifact_spec(stage)
    if spec is None:
        raise ArtifactError("V0 没有需要保存的 Markdown 文件。")
    before = _read_lesson_artifact(spec)
    if before["digest"] != expected_digest:
        raise ArtifactConflictError(
            f"{before['path']} 在编辑期间已被其他页面修改。"
            "请重新读取文件，确认新内容后再保存。"
        )
    if not isinstance(content, str) or not content.strip():
        raise ArtifactError(
            f"{before['path']} 内容不能为空。请填写 Markdown 后重试。"
        )

    clean_content = content.strip()
    payload = f"{clean_content}\n"
    if len(payload.encode("utf-8")) > _MAX_LESSON_ARTIFACT_BYTES:
        raise ArtifactError(
            f"{before['path']} 超过 256 KB。请缩短课程 Markdown 后重试。"
        )
    if clean_content == before["content"]:
        return {
            "before": before,
            "after": before.copy(),
            "changed": False,
            "diff": "",
        }

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=spec.path.parent,
            prefix=f".{spec.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temp_path = Path(temporary.name)

        read_markdown(temp_path)
        if spec.kind == "skill":
            read_skill(temp_path)
        os.replace(temp_path, spec.path)
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError(
            f"无法保存 {before['path']}，原文件保持不变。"
            "请确认文件没有被占用后重试。"
        ) from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    after = _read_lesson_artifact(spec)
    return {
        "before": before,
        "after": after,
        "changed": True,
        "diff": _artifact_diff(before, after),
    }


def get_app_status() -> AppStatus:
    """检查首页运行条件，且不创建模型或发起外部请求。"""

    missing_files = [
        relative_path
        for relative_path in _REQUIRED_APP_FILES
        if not (PROJECT_ROOT / relative_path).is_file()
    ]
    expected_python_version: str | None = None
    version_path = PROJECT_ROOT / ".python-version"
    if version_path.is_file():
        try:
            expected_python_version = version_path.read_text(encoding="utf-8").strip()
        except OSError:
            expected_python_version = None

    python_version = platform.python_version()
    runtime_errors: list[str] = []
    if expected_python_version:
        expected_parts = expected_python_version.split(".")
        try:
            expected_series = tuple(int(part) for part in expected_parts[:2])
        except ValueError:
            expected_series = ()

        current_parts = python_version.split(".")
        try:
            current_series = tuple(int(part) for part in current_parts[:2])
        except ValueError:
            current_series = ()

        if len(expected_series) != 2:
            runtime_errors.append(
                ".python-version 中的已验证 Python 版本无法识别，"
                "请恢复课程文件。"
            )
        elif current_series != expected_series:
            required_series = f"{expected_series[0]}.{expected_series[1]}.x"
            runtime_errors.append(
                "当前 Python 版本为 "
                f"{python_version}，课程要求 Python {required_series}。"
                f"仓库已验证版本为 {expected_python_version}。"
            )

    model_provider: str | None = None
    model_error: str | None = None
    try:
        model_provider = validate_model_configuration()
    except ModelConfigurationError as exc:
        model_error = str(exc)

    if missing_files:
        runtime_errors.append("课程运行所需文件不完整。")

    runtime_ready = not runtime_errors
    model_ready = model_error is None
    errors = list(runtime_errors)
    if model_error is not None:
        errors.append(model_error)

    return {
        "ready": runtime_ready and model_ready,
        "runtime_ready": runtime_ready,
        "model_ready": model_ready,
        "python_version": python_version,
        "expected_python_version": expected_python_version,
        "model_provider": model_provider,
        "missing_files": missing_files,
        "runtime_errors": runtime_errors,
        "model_error": model_error,
        "errors": errors,
    }


def get_model_configuration() -> ModelConfigurationSummary:
    """返回不包含 API Key 的模型配置摘要。"""

    return _get_model_configuration_summary()


def save_model_configuration(
    provider: str,
    api_key: str,
) -> ModelConfigurationSummary:
    """保存首页模型配置，但不发起连接测试。"""

    return _save_model_configuration(provider, api_key)


def test_model_connection() -> AgentResult:
    """使用 V0 发起一次最小真实模型连接测试。"""

    return invoke_v0("请只回复：连接成功")


def chat_v4(
    message: str,
    thread_id: str,
    *,
    attachment: ChatAttachment | None = None,
) -> AgentResult:
    """启动或恢复由 LangGraph checkpoint 管理的 V4 长对话。"""

    if attachment is None:
        return _chat_v4(message, thread_id)
    return _chat_v4(message, thread_id, attachment=attachment)


def invoke(
    stage: str,
    message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
    attachment: ChatAttachment | None = None,
) -> AgentResult:
    """调用 V0 到 V3，V1-V3 可接收由界面保存的完整对话历史。"""

    normalized_stage = stage.strip().upper()
    if normalized_stage == "V0":
        if attachment is not None:
            return invoke_v0(message, attachment=attachment)
        return invoke_v0(message)
    if normalized_stage == "V1":
        if attachment is not None:
            return invoke_v1(message, history=history, attachment=attachment)
        return invoke_v1(message, history=history)
    if normalized_stage == "V2":
        if attachment is not None:
            return invoke_v2(message, history=history, attachment=attachment)
        return invoke_v2(message, history=history)
    if normalized_stage == "V3":
        if attachment is not None:
            return invoke_v3(message, history=history, attachment=attachment)
        return invoke_v3(message, history=history)

    display_stage = normalized_stage or "UNKNOWN"
    return new_agent_result(
        display_stage,
        error=(
            "当前版本仅支持 V0、V1、V2 和 V3，"
            f"收到的阶段为 {display_stage}。"
        ),
    )
