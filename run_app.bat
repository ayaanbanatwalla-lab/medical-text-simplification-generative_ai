@echo off
REM One-click launcher for the Medical Text Simplification Gradio demo.
REM Place this file directly inside the "app" folder and double-click it.

cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo Could not find venv\Scripts\activate.bat
    echo Make sure this file is inside the "app" folder, next to app.py and venv\
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting the app... a browser link will appear below shortly.
echo Keep this window open while you're using the app.
echo Press Ctrl+C here to stop it when you're done.
echo.

python app.py

pause
