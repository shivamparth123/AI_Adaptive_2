@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo The virtual environment was not found. Run setup.ps1 first ^(see README.md^).
    pause
    exit /b 1
)
cd app
"..\venv\Scripts\python.exe" maincode.py
pause
