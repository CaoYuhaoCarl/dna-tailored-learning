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
