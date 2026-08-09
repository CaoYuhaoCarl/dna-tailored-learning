from pathlib import Path

import pytest

from src.storage import FileAlreadyExistsError, StorageError, save_markdown


def test_save_markdown_writes_utf8_file_inside_allowed_root(tmp_path: Path) -> None:
    student_root = tmp_path / "student"
    output_dir = student_root / "reports"

    saved_path = save_markdown(
        output_dir,
        "report.md",
        "\n# 学习报告\n\n今天掌握了现在完成时。\n",
        allowed_root=student_root,
    )

    assert saved_path == output_dir / "report.md"
    assert saved_path.read_text(encoding="utf-8") == (
        "# 学习报告\n\n今天掌握了现在完成时。\n"
    )


@pytest.mark.parametrize(
    "filename",
    ["../escape.md", "nested/report.md", "/tmp/report.md", "report.txt"],
)
def test_save_markdown_rejects_unsafe_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    student_root = tmp_path / "student"

    with pytest.raises(StorageError):
        save_markdown(
            student_root / "reports",
            filename,
            "内容",
            allowed_root=student_root,
        )


def test_save_markdown_rejects_output_directory_outside_allowed_root(
    tmp_path: Path,
) -> None:
    student_root = tmp_path / "student"

    with pytest.raises(StorageError, match="只能写入"):
        save_markdown(
            tmp_path / "outside",
            "report.md",
            "内容",
            allowed_root=student_root,
        )


def test_save_markdown_rejects_symlink_that_escapes_allowed_root(
    tmp_path: Path,
) -> None:
    student_root = tmp_path / "student"
    outside = tmp_path / "outside"
    student_root.mkdir()
    outside.mkdir()
    linked_output = student_root / "linked"
    linked_output.symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageError, match="只能写入"):
        save_markdown(
            linked_output,
            "report.md",
            "内容",
            allowed_root=student_root,
        )

    assert not (outside / "report.md").exists()


def test_save_markdown_never_overwrites_existing_file(tmp_path: Path) -> None:
    student_root = tmp_path / "student"
    output_dir = student_root / "reports"
    saved_path = save_markdown(
        output_dir,
        "report.md",
        "第一版",
        allowed_root=student_root,
    )

    with pytest.raises(FileAlreadyExistsError, match="已存在"):
        save_markdown(
            output_dir,
            "report.md",
            "第二版",
            allowed_root=student_root,
        )

    assert saved_path.read_text(encoding="utf-8") == "第一版\n"
