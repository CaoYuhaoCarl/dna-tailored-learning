from pathlib import Path

import pytest

from src.artifacts import (
    ArtifactError,
    PROMPT_PATH,
    SKILLS_PATH,
    discover_skills,
    read_markdown,
    read_skill,
)


def test_default_student_prompt_contains_socratic_rules() -> None:
    prompt = read_markdown(PROMPT_PATH)

    assert "一次只能问一个简短问题" in prompt
    assert "不直接给出填空答案或完整答案" in prompt


def test_default_student_prompt_routes_non_exercises_to_normal_mode() -> None:
    prompt = read_markdown(PROMPT_PATH)

    assert "最高优先级：先判断回答模式" in prompt
    assert "普通模式" in prompt
    assert "不得编造题目" in prompt
    assert "不强制提问" in prompt
    assert "不回答与学习无关的问题" not in prompt


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


def test_default_skill_uses_standard_metadata_and_steps() -> None:
    skills = discover_skills(SKILLS_PATH)

    assert [skill.name for skill in skills] == ["sorting-out-mistakes"]
    assert "整理错题" in skills[0].description
    assert "原题" in skills[0].description
    artifact = read_skill(skills[0].path)
    assert "Step 1" in artifact.instructions
    assert "student/mistakes/inbox/" in artifact.instructions
    assert "student/mistakes/records/" in artifact.instructions
    assert "save_mistake" in artifact.instructions
    assert "schema_version" in artifact.instructions
    assert "next_review_at" in artifact.instructions


def test_read_skill_requires_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "sorting-out-mistakes"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("# 没有 frontmatter", encoding="utf-8")

    with pytest.raises(ArtifactError, match="缺少 YAML frontmatter"):
        read_skill(skill_path)


def test_read_skill_requires_name_to_match_parent_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "sorting-out-mistakes"
    skill_dir.mkdir()
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: another-skill\n"
        "description: 测试 Skill\n"
        "---\n"
        "\n"
        "Step 1：测试。\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="必须与父目录 sorting-out-mistakes 一致"):
        read_skill(skill_path)


def test_discover_skills_reloads_metadata_after_edit(tmp_path: Path) -> None:
    skills_path = tmp_path / "skill"
    skill_dir = skills_path / "sorting-out-mistakes"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        "description: 第一版触发说明\n"
        "---\n"
        "\n"
        "Step 1：第一版。\n",
        encoding="utf-8",
    )

    first = discover_skills(skills_path)
    skill_path.write_text(
        "---\n"
        "name: sorting-out-mistakes\n"
        "description: 第二版触发说明\n"
        "---\n"
        "\n"
        "Step 1：第二版。\n",
        encoding="utf-8",
    )
    second = discover_skills(skills_path)

    assert first[0].description == "第一版触发说明"
    assert second[0].description == "第二版触发说明"
