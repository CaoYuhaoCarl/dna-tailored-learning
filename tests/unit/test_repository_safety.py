import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_sensitive_files_are_ignored() -> None:
    ignored_entries = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".env" in ignored_entries
    assert "*.log" in ignored_entries
    assert ".ipynb_checkpoints/" in ignored_entries
    assert ".streamlit/secrets.toml" in ignored_entries


def test_teacher_environment_uses_jupyterlab_only() -> None:
    dependencies = {
        line.strip().split("==", maxsplit=1)[0].lower()
        for line in (PROJECT_ROOT / "requirements-dev.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-r"))
    }

    assert "jupyterlab" in dependencies
    assert "notebook" not in dependencies


def test_teacher_notebooks_have_no_saved_outputs() -> None:
    notebook_paths = sorted((PROJECT_ROOT / "teacher").glob("lesson_*.ipynb"))

    assert notebook_paths
    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert code_cells, notebook_path
        assert all(cell["execution_count"] is None for cell in code_cells), notebook_path
        assert all(cell["outputs"] == [] for cell in code_cells), notebook_path


def test_mistake_inbox_and_normalized_records_are_separated() -> None:
    mistakes_root = PROJECT_ROOT / "student" / "mistakes"
    inbox_files = sorted((mistakes_root / "inbox").glob("*.md"))
    record_files = sorted((mistakes_root / "records").glob("*/*.md"))

    assert not list(mistakes_root.glob("*.md"))
    assert inbox_files
    assert record_files
    assert all(path.name.startswith("mistake-") for path in record_files)
    for path in record_files:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "schema_version: 1" in content
        assert f"id: {path.stem}" in content
        assert f"subject: {path.parent.name}" in content
        assert "topic: " in content
        assert "status: needs-review" in content
        assert "created_at: \"" in content
        assert "review_count: 0" in content
        assert "next_review_at: null" in content
        assert 'source: "inbox/' in content
        assert "# 错题记录" in content


def test_teacher_v1_dialogue_starts_from_live_student_input() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_1_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definition = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def run_v1_dialogue" in "".join(cell["source"])
    )
    launcher = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "v1_dialogue_history = run_v1_dialogue()" in "".join(cell["source"])
    )

    assert 'student_message = input("学生：")' in definition
    assert 'invoke("V1", student_message, history=history)' in definition
    assert "ask_v1(question)" not in definition
    assert "START_INTERACTIVE_V1" not in launcher
    assert launcher.strip() == "v1_dialogue_history = run_v1_dialogue()"


def test_teacher_v1_dialogue_passes_each_student_input_with_history() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_1_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definition_cell = next(
        cell
        for cell in notebook["cells"]
        if "def run_v1_dialogue" in "".join(cell["source"])
    )
    student_inputs = iter(["我的题目", "我的回答", "/exit"])
    calls = []

    def fake_input(prompt: str) -> str:
        assert prompt == "学生："
        return next(student_inputs)

    def fake_invoke(stage: str, message: str, *, history: list[dict]) -> dict:
        calls.append((stage, message, [item.copy() for item in history]))
        return {"text": f"针对{message}的一个问题？", "error": None}

    namespace = {"input": fake_input, "invoke": fake_invoke}
    exec("".join(definition_cell["source"]), namespace)

    history = namespace["run_v1_dialogue"]()

    assert calls == [
        ("V1", "我的题目", []),
        (
            "V1",
            "我的回答",
            [
                {"role": "user", "content": "我的题目"},
                {"role": "assistant", "content": "针对我的题目的一个问题？"},
            ],
        ),
    ]
    assert history[-2:] == [
        {"role": "user", "content": "我的回答"},
        {"role": "assistant", "content": "针对我的回答的一个问题？"},
    ]


def test_teacher_v2_dialogue_uses_live_input_and_displays_loaded_skill() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_2_skill.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definition_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["id"] == "v2-dialogue-definition"
    )
    definition = "".join(definition_cell["source"])
    assert 'call.get("name") == "load_skill"' in definition
    assert 'call.get("name") == "load_mistake_file"' in definition
    assert 'call.get("name") == "save_mistake"' in definition
    student_inputs = iter(["请帮我整理错题", "/exit"])
    calls = []

    def fake_input(prompt: str) -> str:
        assert prompt == "学生："
        return next(student_inputs)

    def fake_invoke(stage: str, message: str, *, history: list[dict]) -> dict:
        calls.append((stage, message, [item.copy() for item in history]))
        return {
            "text": "错题已经保存。",
            "tool_calls": [
                {
                    "name": "load_skill",
                    "args": {"skill_name": "sorting-out-mistakes"},
                },
                {
                    "name": "load_mistake_file",
                    "args": {"path": "student/mistakes/inbox/english.md"},
                },
                {
                    "name": "save_mistake",
                    "args": {"original_question": "测试题"},
                },
            ],
            "error": None,
        }

    namespace = {"input": fake_input, "invoke": fake_invoke}
    exec("".join(definition_cell["source"]), namespace)

    session = namespace["run_v2_dialogue"]()

    assert calls == [("V2", "请帮我整理错题", [])]
    assert session["history"] == [
        {"role": "user", "content": "请帮我整理错题"},
        {"role": "assistant", "content": "错题已经保存。"},
    ]
    assert session["tool_calls"][0]["name"] == "load_skill"
    assert session["tool_calls"][1]["name"] == "load_mistake_file"
    assert session["tool_calls"][2]["name"] == "save_mistake"
