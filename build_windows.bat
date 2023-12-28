@echo off

where python > nul 2>nul
if %errorlevel% neq 0 (
    echo Python version 3 or higher must be installed before run
) else (
    python --version

    pip install -r requirements.txt
)