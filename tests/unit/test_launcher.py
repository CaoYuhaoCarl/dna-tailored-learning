from io import StringIO
from pathlib import Path
import stat
import sys

import pytest

from scripts import launch_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("3.14.0", (3, 14)),
        ("3.14.99", (3, 14)),
        ("3.14", (3, 14)),
        ("3", None),
        ("unknown", None),
    ],
)
def test_parse_python_series(
    version: str,
    expected: tuple[int, int] | None,
) -> None:
    assert launch_app.parse_python_series(version) == expected


def test_required_file_check_explains_that_zip_must_be_extracted(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("", encoding="utf-8")

    with pytest.raises(launch_app.LaunchError, match="完整解压") as error:
        launch_app.validate_required_files(tmp_path)

    assert ".env" in str(error.value)
    assert "requirements.txt" in str(error.value)


def test_main_rejects_missing_env_before_creating_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for relative_path in launch_app.REQUIRED_FILES:
        if relative_path == ".env":
            continue
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if relative_path == ".python-version" else "ok\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(launch_app, "PROJECT_ROOT", tmp_path)

    exit_code = launch_app.main(["--check-only"])

    assert exit_code == 1
    assert not (tmp_path / ".venv").exists()
    log = (tmp_path / "logs" / "startup.log").read_text(encoding="utf-8")
    assert "缺少文件" in log
    assert ".env" in log


def test_requirements_digest_controls_dependency_reuse(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    marker = tmp_path / ".venv" / ".requirements.sha256"
    marker.parent.mkdir()
    requirements.write_text("streamlit==1.60.0\n", encoding="utf-8")

    assert launch_app.dependencies_need_install(requirements, marker) is True

    marker.write_text(
        launch_app.requirements_digest(requirements) + "\n",
        encoding="utf-8",
    )

    assert launch_app.dependencies_need_install(requirements, marker) is False

    requirements.write_text("streamlit==1.60.1\n", encoding="utf-8")

    assert launch_app.dependencies_need_install(requirements, marker) is True


@pytest.mark.parametrize(
    "macos_version",
    ["12.0", "12.7.6", "15.6.1", "26.0"],
)
def test_supported_platform_accepts_macos_12_or_newer(
    macos_version: str,
) -> None:
    launch_app.validate_supported_platform("darwin", macos_version)


def test_supported_platform_ignores_non_macos() -> None:
    launch_app.validate_supported_platform("win32", "unknown")


@pytest.mark.parametrize("macos_version", ["11.7.10", "unknown", ""])
def test_supported_platform_rejects_unsupported_or_unknown_macos(
    macos_version: str,
) -> None:
    with pytest.raises(launch_app.LaunchError, match="macOS 12"):
        launch_app.validate_supported_platform("darwin", macos_version)


def test_existing_invalid_venv_is_not_deleted(tmp_path: Path) -> None:
    venv_path = tmp_path / ".venv"
    venv_path.mkdir()
    sentinel = venv_path / "keep-me.txt"
    sentinel.write_text("user data", encoding="utf-8")

    with pytest.raises(launch_app.LaunchError, match="请删除项目中的 .venv"):
        launch_app.prepare_environment(
            tmp_path,
            (3, 14),
            log=StringIO(),
            environment={},
        )

    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_fresh_environment_creates_venv_and_installs_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("streamlit==1.60.0\n", encoding="utf-8")
    python_path = tmp_path / ".venv" / "bin" / "python"
    calls: list[tuple[str, list[str], dict[str, str]]] = []
    monkeypatch.setattr(launch_app, "_venv_python", lambda _root: python_path)
    monkeypatch.setattr(
        launch_app,
        "_validate_venv_python",
        lambda *args, **kwargs: None,
    )

    def record_command(command, *, description: str, environment, **kwargs) -> None:
        calls.append((description, list(command), dict(environment)))
        if description == "创建项目虚拟环境":
            python_path.parent.mkdir(parents=True)

    monkeypatch.setattr(launch_app, "run_logged_command", record_command)

    selected_python = launch_app.prepare_environment(
        tmp_path,
        (3, 14),
        log=StringIO(),
        environment={"PIP_INDEX_URL": launch_app.DEFAULT_PIP_INDEX_URL},
    )

    assert selected_python == python_path
    assert [description for description, _, _ in calls] == [
        "创建项目虚拟环境",
        "安装课程运行依赖",
        "检查依赖兼容性",
        "检查核心模块",
    ]
    _, install_command, install_environment = calls[1]
    assert "--only-binary=:all:" in install_command
    assert (
        install_environment["PIP_INDEX_URL"]
        == launch_app.DEFAULT_PIP_INDEX_URL
    )
    marker = tmp_path / ".venv" / launch_app.REQUIREMENTS_MARKER
    assert marker.read_text(encoding="utf-8").strip() == (
        launch_app.requirements_digest(requirements)
    )


def test_dependency_marker_reuses_existing_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("streamlit==1.60.0\n", encoding="utf-8")
    venv_path = tmp_path / ".venv"
    venv_path.mkdir()
    marker = venv_path / launch_app.REQUIREMENTS_MARKER
    marker.write_text(
        launch_app.requirements_digest(requirements) + "\n",
        encoding="utf-8",
    )
    python_path = tmp_path / "fake-python"
    calls: list[str] = []
    monkeypatch.setattr(launch_app, "_venv_python", lambda _root: python_path)
    monkeypatch.setattr(
        launch_app,
        "_validate_venv_python",
        lambda *args, **kwargs: None,
    )

    def record_command(*args, description: str, **kwargs) -> None:
        calls.append(description)

    monkeypatch.setattr(launch_app, "run_logged_command", record_command)

    selected_python = launch_app.prepare_environment(
        tmp_path,
        (3, 14),
        log=StringIO(),
        environment={},
    )

    assert selected_python == python_path
    assert calls == ["检查现有运行环境"]


def test_failed_dependency_install_does_not_write_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("streamlit==1.60.0\n", encoding="utf-8")
    marker = tmp_path / ".venv" / launch_app.REQUIREMENTS_MARKER

    def fail_install(*args, description: str, **kwargs) -> None:
        raise launch_app.LaunchError(f"{description}失败")

    monkeypatch.setattr(launch_app, "run_logged_command", fail_install)

    with pytest.raises(launch_app.LaunchError, match="安装课程运行依赖失败"):
        launch_app._install_dependencies(
            tmp_path,
            Path(sys.executable),
            requirements,
            marker,
            log=StringIO(),
            environment={},
        )

    assert not marker.exists()


def test_streamlit_command_uses_private_loopback_and_no_prompts(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "中文 课程"
    python = project_root / ".venv" / "bin" / "python"

    command = launch_app.streamlit_command(project_root, python)

    assert command[:5] == [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(project_root / "app.py"),
    ]
    assert "--server.address=127.0.0.1" in command
    assert "--server.headless=false" in command
    assert "--server.showEmailPrompt=false" in command
    assert "--browser.gatherUsageStats=false" in command
    assert "--logger.hideWelcomeMessage=true" in command


def test_logged_command_preserves_output_and_failure(tmp_path: Path) -> None:
    log = StringIO()
    secret = "must-not-appear"

    with pytest.raises(launch_app.LaunchError, match="测试命令失败"):
        launch_app.run_logged_command(
            [
                sys.executable,
                "-c",
                "import sys; print('visible output'); raise SystemExit(7)",
            ],
            cwd=tmp_path,
            log=log,
            description="测试命令",
            environment={"SECRET_FOR_TEST": secret},
        )

    contents = log.getvalue()
    assert "visible output" in contents
    assert secret not in contents


def test_main_does_not_copy_api_key_into_failure_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "super-secret-api-key"
    for relative_path in launch_app.REQUIRED_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == ".python-version":
            content = "3.14.3\n"
        elif relative_path == ".env":
            content = f"MODEL_PROVIDER=deepseek\nDEEPSEEK_API_KEY={secret}\n"
        else:
            content = "ok\n"
        path.write_text(content, encoding="utf-8")

    monkeypatch.setattr(launch_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        launch_app,
        "prepare_environment",
        lambda *args, **kwargs: Path(sys.executable),
    )
    monkeypatch.setattr(
        launch_app,
        "run_logged_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            launch_app.LaunchError("MODEL_PROVIDER 配置无效。")
        ),
    )

    exit_code = launch_app.main(["--check-only"])

    assert exit_code == 1
    log = (tmp_path / "logs" / "startup.log").read_text(encoding="utf-8")
    assert "MODEL_PROVIDER 配置无效" in log
    assert "修复步骤" in log
    assert secret not in log


@pytest.mark.parametrize(
    ("configured_index", "expected_index"),
    [
        (None, launch_app.DEFAULT_PIP_INDEX_URL),
        (
            "https://mirror.example.test/simple",
            "https://mirror.example.test/simple",
        ),
    ],
)
def test_main_defaults_to_tuna_and_preserves_pip_index_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_index: str | None,
    expected_index: str,
) -> None:
    for relative_path in launch_app.REQUIRED_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if relative_path == ".python-version" else "ok\n",
            encoding="utf-8",
        )

    if configured_index is None:
        monkeypatch.delenv("PIP_INDEX_URL", raising=False)
    else:
        monkeypatch.setenv("PIP_INDEX_URL", configured_index)

    captured_environment: dict[str, str] = {}

    def capture_environment(*args, environment, **kwargs) -> Path:
        captured_environment.update(environment)
        return Path(sys.executable)

    monkeypatch.setattr(launch_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launch_app, "_validate_host_python", lambda *args: None)
    monkeypatch.setattr(
        launch_app,
        "validate_supported_platform",
        lambda *args: None,
    )
    monkeypatch.setattr(launch_app, "prepare_environment", capture_environment)
    monkeypatch.setattr(launch_app, "run_logged_command", lambda *args, **kwargs: None)

    exit_code = launch_app.main(["--check-only"])

    assert exit_code == 0
    assert captured_environment["PIP_INDEX_URL"] == expected_index


def test_main_rejects_old_macos_before_preparing_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for relative_path in launch_app.REQUIRED_FILES:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "3.14.3\n" if relative_path == ".python-version" else "ok\n",
            encoding="utf-8",
        )

    prepare_environment_called = False

    def reject_macos(*args) -> None:
        raise launch_app.LaunchError("课程需要 macOS 12 或更高版本。")

    def record_prepare(*args, **kwargs) -> Path:
        nonlocal prepare_environment_called
        prepare_environment_called = True
        return Path(sys.executable)

    monkeypatch.setattr(launch_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launch_app, "_validate_host_python", lambda *args: None)
    monkeypatch.setattr(launch_app, "validate_supported_platform", reject_macos)
    monkeypatch.setattr(launch_app, "prepare_environment", record_prepare)

    exit_code = launch_app.main(["--check-only"])

    assert exit_code == 1
    assert prepare_environment_called is False
    assert not (tmp_path / ".venv").exists()
    log = (tmp_path / "logs" / "startup.log").read_text(encoding="utf-8")
    assert "macOS 12" in log


def test_launch_scripts_have_expected_entrypoints_and_permissions() -> None:
    mac_script = PROJECT_ROOT / "start_mac.sh"
    mac_command = PROJECT_ROOT / "start_mac.command"
    windows_script = PROJECT_ROOT / "start_windows.bat"

    assert mac_script.stat().st_mode & stat.S_IXUSR
    assert mac_command.stat().st_mode & stat.S_IXUSR
    assert '"$SCRIPT_DIR/scripts/launch_app.py"' in mac_script.read_text(
        encoding="utf-8"
    )
    assert '"$SCRIPT_DIR/start_mac.sh"' in mac_command.read_text(encoding="utf-8")
    windows_source = windows_script.read_text(encoding="utf-8")
    assert "PYTHON_MANAGER_AUTOMATIC_INSTALL=false" in windows_source
    assert '"%~dp0scripts\\launch_app.py"' in windows_source
