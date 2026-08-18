@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON_MANAGER_AUTOMATIC_INSTALL=false"
set "PYTHONUTF8=1"

py -V:3.14 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)" >nul 2>&1
if not errorlevel 1 goto run_python_manager

py -3.14 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)" >nul 2>&1
if not errorlevel 1 goto run_legacy_launcher

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 14) else 1)" >nul 2>&1
if not errorlevel 1 goto run_python

echo 启动失败：未找到 Python 3.14.x。
echo 请先安装 Python 3.14，然后重新双击 start_windows.bat。
set "EXIT_CODE=1"
goto finish

:run_python_manager
py -V:3.14 "%~dp0scripts\launch_app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_legacy_launcher
py -3.14 "%~dp0scripts\launch_app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_python
python "%~dp0scripts\launch_app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

:finish
if "%EXIT_CODE%"=="0" exit /b 0
if "%EXIT_CODE%"=="130" exit /b 130
echo.
pause
exit /b %EXIT_CODE%
