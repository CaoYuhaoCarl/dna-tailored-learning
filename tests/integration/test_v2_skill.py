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
    mistakes_path = student_root / "mistakes"
    monkeypatch.setattr(agents_module, "STUDENT_ROOT", student_root)
    monkeypatch.setattr(agents_module, "MISTAKES_PATH", mistakes_path)

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
    assert len(list(mistakes_path.glob("mistake-*.md"))) == 1

    unrelated = invoke("V2", "请用一句简短的中文和我打招呼。")

    assert unrelated["error"] is None, unrelated["error"]
    assert unrelated["text"].strip()
    assert not [
        call
        for call in unrelated["tool_calls"]
        if call.get("name") == "load_skill"
    ]
