@echo off
cd /d "%~dp0"
echo Starting setup in: %cd%
if not exist "venv\Scripts\python.exe" (
    py -m venv venv
    if errorlevel 1 python -m venv venv
)
if not exist "venv\Scripts\python.exe" (
    echo Python was not found. Install Python 3.10 to 3.12 from python.org and tick Add to PATH.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Install failed. Read the message above.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -m streamlit run app.py
pause
