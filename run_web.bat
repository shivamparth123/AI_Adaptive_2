@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo The virtual environment was not found. Run setup.ps1 first ^(see README.md^).
    pause
    exit /b 1
)
echo.
echo  Adaptive Local AI - web app
echo  Make sure Ollama is running ^(tray icon^).
echo  Open  http://127.0.0.1:8000  in your browser.   Press Ctrl+C here to stop.
echo.
cd webapp\backend
"..\..\venv\Scripts\python.exe" -m uvicorn app:app --port 8000
pause
