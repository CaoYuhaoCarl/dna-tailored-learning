"""学生数据目录下可复用的安全 Markdown 写入逻辑。"""

from pathlib import Path

from src.artifacts import PROJECT_ROOT


STUDENT_ROOT = PROJECT_ROOT / "student"
MISTAKES_PATH = STUDENT_ROOT / "mistakes"


class StorageError(ValueError):
    """文件内容或写入位置不符合学生数据目录规则。"""


class FileAlreadyExistsError(StorageError):
    """目标文件已存在，拒绝静默覆盖。"""


def save_markdown(
    output_dir: str | Path,
    filename: str,
    content: str,
    *,
    allowed_root: str | Path = STUDENT_ROOT,
) -> Path:
    """在允许目录内新建 UTF-8 Markdown 文件，且不覆盖已有文件。"""

    clean_content = content.strip()
    if not clean_content:
        raise StorageError("Markdown 内容为空，未写入文件。")

    clean_filename = filename.strip()
    filename_path = Path(clean_filename)
    if (
        not clean_filename
        or filename_path.is_absolute()
        or len(filename_path.parts) != 1
        or filename_path.name != clean_filename
    ):
        raise StorageError("文件名只能包含名称本身，不能包含目录或绝对路径。")
    if filename_path.suffix.lower() != ".md":
        raise StorageError("文件名必须使用 .md 扩展名。")

    root_path = Path(allowed_root).resolve()
    directory_path = Path(output_dir).resolve()
    try:
        directory_path.relative_to(root_path)
    except ValueError as exc:
        raise StorageError(f"只能写入 {root_path} 目录及其子目录。") from exc

    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        directory_path = directory_path.resolve()
        target_path = (directory_path / clean_filename).resolve()
        target_path.relative_to(directory_path)
        with target_path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(f"{clean_content}\n")
    except FileExistsError as exc:
        raise FileAlreadyExistsError(
            f"{target_path} 已存在，未覆盖原文件。"
        ) from exc
    except ValueError as exc:
        raise StorageError("目标文件解析到了允许目录之外，未写入文件。") from exc
    except OSError as exc:
        raise StorageError(
            f"无法写入 {directory_path}。请检查目录权限和磁盘空间。"
        ) from exc

    return target_path
