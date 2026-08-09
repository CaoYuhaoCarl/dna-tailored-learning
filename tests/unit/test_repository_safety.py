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


def test_teacher_notebook_has_no_saved_outputs() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_1_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_teacher_v1_dialogue_starts_from_live_student_input() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_1_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells_by_id = {cell["id"]: cell for cell in notebook["cells"]}
    definition = "".join(cells_by_id["v1-dialogue-definition"]["source"])
    launcher = "".join(cells_by_id["v1-dialogue-loop"]["source"])

    assert 'student_message = input("学生：")' in definition
    assert 'invoke("V1", student_message, history=history)' in definition
    assert "ask_v1(question)" not in definition
    assert "START_INTERACTIVE_V1" not in launcher
    assert launcher.strip() == "v1_dialogue_history = run_v1_dialogue()"


def test_teacher_v1_dialogue_passes_each_student_input_with_history() -> None:
    notebook_path = PROJECT_ROOT / "teacher" / "lesson_1_demo.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definition_cell = next(
        cell for cell in notebook["cells"] if cell["id"] == "v1-dialogue-definition"
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
