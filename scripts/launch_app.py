"""为 Windows 和 macOS 启动脚本提供统一的安装与启动流程。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    ".python-version",
    ".env",
    "requirements.txt",
    "app.py",
    "src/model.py",
)
REQUIREMENTS_MARKER = ".requirements.sha256"
DEPENDENCY_IMPORT_CHECK = "import streamlit; import src.facade"
APPLICATION_CHECK = """
from src.facade import get_app_status

status = get_app_status()
if not status["ready"]:
    print("运行条件检查失败：")
    for error in status["errors"]:
        print(f"- {error}")
    raise SystemExit(1)
print(f"运行环境检查通过，模型供应商：{status['model_provider']}。")
""".strip()


class LaunchError(RuntimeError):
    """启动准备或应用进程失败。"""


def parse_python_series(version: str) -> tuple[int, int] | None:
    """将 Python 版本解析为 major/minor，无法解析时返回 None。"""

    parts = version.strip().split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def validate_required_files(project_root: Path) -> None:
    """确认源码 ZIP 已完整解压，并且本地配置文件已经创建。"""

    missing = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (project_root / relative_path).is_file()
    ]
    if not missing:
        return

    missing_list = "\n".join(f"- {path}" for path in missing)
    raise LaunchError(
        "启动文件不完整。请先完整解压源码 ZIP，并按 README 配置 .env。"
        f"\n缺少文件：\n{missing_list}"
    )


def requirements_digest(requirements_path: Path) -> str:
    """返回依赖文件的稳定摘要。"""

    return sha256(requirements_path.read_bytes()).hexdigest()


def dependencies_need_install(
    requirements_path: Path,
    marker_path: Path,
) -> bool:
    """判断依赖文件是否在当前虚拟环境中完成过安装。"""

    try:
        installed_digest = marker_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return True
    return installed_digest != requirements_digest(requirements_path)


def _emit(log: TextIO, message: str) -> None:
    print(message, flush=True)
    print(message, file=log, flush=True)


def _emit_repair_steps(log: TextIO, log_path: Path) -> None:
    _emit(log, "修复步骤：")
    _emit(log, "1. 确认源码 ZIP 已完整解压，并已手动创建和填写 .env。")
    _emit(log, "2. 首次安装依赖时保持网络连接，然后重新启动。")
    _emit(log, "3. 如果提示 .venv 损坏，请删除项目中的 .venv 文件夹后重试。")
    _emit(log, f"4. 如果仍然失败，请保留此窗口并查看日志：{log_path}")


def run_logged_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log: TextIO,
    description: str,
    environment: Mapping[str, str] | None = None,
) -> None:
    """运行命令，同时将输出写到终端和本次启动日志。"""

    _emit(log, f"{description}...")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise LaunchError(f"无法执行{description}：{exc}") from exc

    try:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    if return_code in {130, -2, -1073741510}:
        raise KeyboardInterrupt
    if return_code != 0:
        raise LaunchError(f"{description}失败，退出码为 {return_code}。")


def _expected_python(project_root: Path) -> tuple[str, tuple[int, int]]:
    version_path = project_root / ".python-version"
    try:
        verified_version = version_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise LaunchError(f"无法读取 {version_path.name}：{exc}") from exc

    expected_series = parse_python_series(verified_version)
    if expected_series is None:
        raise LaunchError(
            f".python-version 内容无效：{verified_version!r}。"
            "请重新下载完整源码 ZIP。"
        )
    return verified_version, expected_series


def _validate_host_python(
    verified_version: str,
    expected_series: tuple[int, int],
) -> None:
    current_series = sys.version_info[:2]
    if current_series == expected_series:
        return

    required = f"{expected_series[0]}.{expected_series[1]}.x"
    current = ".".join(str(part) for part in sys.version_info[:3])
    raise LaunchError(
        f"当前 Python 为 {current}，课程需要 Python {required}。"
        f"仓库已验证版本为 {verified_version}，请先安装兼容版本。"
    )


def _venv_python(project_root: Path) -> Path:
    if os.name == "nt":
        return project_root / ".venv" / "Scripts" / "python.exe"
    return project_root / ".venv" / "bin" / "python"


def _validate_venv_python(
    python_path: Path,
    expected_series: tuple[int, int],
    *,
    project_root: Path,
    environment: Mapping[str, str],
) -> None:
    if not python_path.is_file():
        raise LaunchError(
            "现有 .venv 不完整。请删除项目中的 .venv 文件夹后重新启动。"
        )

    try:
        completed = subprocess.run(
            [
                str(python_path),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            cwd=project_root,
            env=dict(environment),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise LaunchError(
            "现有 .venv 无法运行。请删除项目中的 .venv 文件夹后重新启动。"
        ) from exc

    expected = f"{expected_series[0]}.{expected_series[1]}"
    if completed.returncode != 0 or completed.stdout.strip() != expected:
        raise LaunchError(
            f"现有 .venv 不是 Python {expected}.x。"
            "请删除项目中的 .venv 文件夹后重新启动。"
        )


def _install_dependencies(
    project_root: Path,
    python_path: Path,
    requirements_path: Path,
    marker_path: Path,
    *,
    log: TextIO,
    environment: Mapping[str, str],
) -> None:
    run_logged_command(
        [str(python_path), "-m", "pip", "install", "-r", str(requirements_path)],
        cwd=project_root,
        log=log,
        description="安装课程运行依赖",
        environment=environment,
    )
    run_logged_command(
        [str(python_path), "-m", "pip", "check"],
        cwd=project_root,
        log=log,
        description="检查依赖兼容性",
        environment=environment,
    )
    run_logged_command(
        [str(python_path), "-c", DEPENDENCY_IMPORT_CHECK],
        cwd=project_root,
        log=log,
        description="检查核心模块",
        environment=environment,
    )
    marker_path.write_text(
        requirements_digest(requirements_path) + "\n",
        encoding="utf-8",
    )


def prepare_environment(
    project_root: Path,
    expected_series: tuple[int, int],
    *,
    log: TextIO,
    environment: Mapping[str, str],
) -> Path:
    """创建或复用项目虚拟环境，并返回其中的 Python。"""

    venv_path = project_root / ".venv"
    python_path = _venv_python(project_root)
    requirements_path = project_root / "requirements.txt"
    marker_path = venv_path / REQUIREMENTS_MARKER

    if not venv_path.exists():
        run_logged_command(
            [sys.executable, "-m", "venv", str(venv_path)],
            cwd=project_root,
            log=log,
            description="创建项目虚拟环境",
            environment=environment,
        )

    _validate_venv_python(
        python_path,
        expected_series,
        project_root=project_root,
        environment=environment,
    )

    if dependencies_need_install(requirements_path, marker_path):
        _install_dependencies(
            project_root,
            python_path,
            requirements_path,
            marker_path,
            log=log,
            environment=environment,
        )
        return python_path

    try:
        run_logged_command(
            [str(python_path), "-c", DEPENDENCY_IMPORT_CHECK],
            cwd=project_root,
            log=log,
            description="检查现有运行环境",
            environment=environment,
        )
    except LaunchError:
        _emit(log, "现有依赖不完整，将重新安装。")
        _install_dependencies(
            project_root,
            python_path,
            requirements_path,
            marker_path,
            log=log,
            environment=environment,
        )
    return python_path


def streamlit_command(project_root: Path, python_path: Path) -> list[str]:
    """构造只监听本机且不显示首次使用提示的启动命令。"""

    return [
        str(python_path),
        "-m",
        "streamlit",
        "run",
        str(project_root / "app.py"),
        "--server.address=127.0.0.1",
        "--server.headless=false",
        "--server.showEmailPrompt=false",
        "--browser.gatherUsageStats=false",
        "--logger.hideWelcomeMessage=true",
    ]


def _parse_args(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备并启动课程应用。")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查运行环境，不启动 Streamlit。",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    project_root = PROJECT_ROOT
    log_path = project_root / "logs" / "startup.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"无法创建启动日志 {log_path}：{exc}", file=sys.stderr)
        return 1

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INPUT"] = "1"

    with log:
        _emit(log, f"课程启动检查开始：{datetime.now().isoformat(timespec='seconds')}")
        try:
            validate_required_files(project_root)
            verified_version, expected_series = _expected_python(project_root)
            _validate_host_python(verified_version, expected_series)
            python_path = prepare_environment(
                project_root,
                expected_series,
                log=log,
                environment=environment,
            )
            run_logged_command(
                [str(python_path), "-c", APPLICATION_CHECK],
                cwd=project_root,
                log=log,
                description="检查课程文件和模型配置",
                environment=environment,
            )
            if args.check_only:
                _emit(log, "启动检查全部通过。")
                return 0

            _emit(log, "正在启动课程界面，浏览器将自动打开。")
            _emit(log, "关闭此窗口或按 Ctrl+C 可以停止课程应用。")
            run_logged_command(
                streamlit_command(project_root, python_path),
                cwd=project_root,
                log=log,
                description="启动 Streamlit",
                environment=environment,
            )
            return 0
        except KeyboardInterrupt:
            _emit(log, "课程应用已停止。")
            return 130
        except LaunchError as exc:
            _emit(log, f"启动失败：{exc}")
            _emit_repair_steps(log, log_path)
            return 1
        except OSError as exc:
            _emit(log, f"启动失败：{exc}")
            _emit_repair_steps(log, log_path)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
