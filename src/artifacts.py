"""学生可编辑 Markdown 文件的读取与校验。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "student" / "prompt.md"


class ArtifactError(ValueError):
    """学生文件缺失、为空或无法读取。"""


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_markdown(path: str | Path) -> str:
    """使用 UTF-8 读取非空 Markdown，每次调用都重新访问磁盘。"""

    markdown_path = Path(path)
    display_path = _display_path(markdown_path)
    try:
        content = markdown_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactError(
            f"未找到 {display_path}。请从课程包恢复该文件后重试。"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ArtifactError(
            f"无法读取 {display_path}：文件不是 UTF-8 编码。"
            "请将文件另存为 UTF-8 后重试。"
        ) from exc
    except OSError as exc:
        raise ArtifactError(
            f"无法读取 {display_path}。请确认它是可访问的 Markdown 文件后重试。"
        ) from exc

    clean_content = content.strip()
    if not clean_content:
        raise ArtifactError(
            f"{display_path} 内容为空。请填写 Prompt、保存文件后重试。"
        )
    return clean_content
