@echo off
REM Install uv
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo uv installed. Please restart this script.
    pause
    exit /b 0
)

cd /d "%~dp0"
uv run python launcher.py