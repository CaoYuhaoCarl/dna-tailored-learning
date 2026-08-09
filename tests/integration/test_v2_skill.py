import os
from pathlib import Path

import pytest

import src.agents as agents_module
from src.facade import invoke


@pytest.mark.integration
def test_v2_loads_and_saves_skill_for_structured_mistake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.getenv("RUN_DEEPSEEK_INTEGRATION") != "1":
        pytest.skip("设置 RUN_DEEPSEEK_INTEGRATION=1 后才调用真实 DeepSeek。")

    student_root = tmp_path / "student"
    inbox_path = student_root / "mistakes" / "inbox"
    records_path = student_root / "mistakes" / "records"
    monkeypatch.setattr(agents_module, "MISTAKES_INBOX_PATH", inbox_path)
    monkeypatch.setattr(agents_module, "MISTAKES_RECORDS_PATH", records_path)

    matching = invoke(
        "V2",
        "错题1 类型：语法填空 原题：I ____ (read) this book three times. "
        "我的答案：am reading",
    )

    assert matching["error"] is None, matching["error"]
    assert matching["text"].strip()
    loaded_skills = [
        call.get("args", {}).get("skill_name")
        for call in matching["tool_calls"]
        if call.get("name") == "load_skill"
    ]
    assert loaded_skills == ["sorting-out-mistakes"]
    assert [
        call
        for call in matching["tool_calls"]
        if call.get("name") == "save_mistake"
    ]
    saved_files = list((records_path / "english").glob("mistake-*.md"))
    assert len(saved_files) == 1
    saved_content = saved_files[0].read_text(encoding="utf-8")
    assert saved_content.startswith("---\n")
    assert f"id: {saved_files[0].stem}" in saved_content
    assert 'source: "chat"' in saved_content

    unrelated = invoke("V2", "请用一句简短的中文和我打招呼。")

    assert unrelated["error"] is None, unrelated["error"]
    assert unrelated["text"].strip()
    assert not [
        call
        for call in unrelated["tool_calls"]
        if call.get("name") == "load_skill"
    ]


@pytest.mark.integration
def test_v2_reads_file_and_saves_every_mistake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.getenv("RUN_DEEPSEEK_INTEGRATION") != "1":
        pytest.skip("设置 RUN_DEEPSEEK_INTEGRATION=1 后才调用真实 DeepSeek。")

    student_root = tmp_path / "student"
    inbox_path = student_root / "mistakes" / "inbox"
    records_path = student_root / "mistakes" / "records"
    inbox_path.mkdir(parents=True)
    source_path = inbox_path / "english.md"
    source_path.write_text(
        "错题1\n"
        "类型：语法填空\n"
        "原题：I ____ (read) this book three times.\n"
        "我的答案：am reading\n\n"
        "错题2\n"
        "类型：语法填空\n"
        "原题：She ____ (go) to the library yesterday.\n"
        "我的答案：has gone\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agents_module, "MISTAKES_INBOX_PATH", inbox_path)
    monkeypatch.setattr(agents_module, "MISTAKES_RECORDS_PATH", records_path)

    result = invoke("V2", f"继续整理这个文件里的全部错题：{source_path}")

    assert result["error"] is None, result["error"]
    assert [call["name"] for call in result["tool_calls"]].count(
        "load_mistake_file"
    ) == 1
    assert [call["name"] for call in result["tool_calls"]].count(
        "save_mistake"
    ) == 2
    saved_files = list((records_path / "english").glob("mistake-*.md"))
    assert len(saved_files) == 2
    assert all(
        'source: "inbox/english.md"' in path.read_text(encoding="utf-8")
        for path in saved_files
    )
