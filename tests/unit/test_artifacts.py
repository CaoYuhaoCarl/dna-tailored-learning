from pathlib import Path

import pytest

from src.artifacts import ArtifactError, PROMPT_PATH, read_markdown


def test_default_student_prompt_contains_socratic_rules() -> None:
    prompt = read_markdown(PROMPT_PATH)

    assert "一次只能问一个简短问题" in prompt
    assert "不直接给出填空答案或完整答案" in prompt


def test_read_markdown_returns_trimmed_content(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("\n# 测试 Prompt\n\n保留内部空行。\n", encoding="utf-8")

    assert read_markdown(prompt_path) == "# 测试 Prompt\n\n保留内部空行。"


def test_read_markdown_reports_missing_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "missing.md"

    with pytest.raises(ArtifactError, match=r"未找到 .*missing\.md"):
        read_markdown(prompt_path)


def test_read_markdown_reports_empty_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ArtifactError, match=r"prompt\.md 内容为空"):
        read_markdown(prompt_path)


def test_read_markdown_reports_non_utf8_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ArtifactError, match="不是 UTF-8 编码"):
        read_markdown(prompt_path)
