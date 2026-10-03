@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo    Conversational Data Analysis Assistant (LLM Powered) - Launcher
echo ==============================================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ from https://python.org and add it to PATH.
    pause
    exit /b 1
)

:: 2. Check virtual environment
if not exist ".venv" (
    echo [INFO] Virtual environment not found. Creating .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
)

:: 3. Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 4. Install / Update dependencies
echo [INFO] Checking and installing dependencies from requirements.txt...
python -m pip install --quiet --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install required packages!
    pause
    exit /b 1
)

:: 5. Check .env configuration file
if not exist ".env" (
    echo [WARNING] .env file not found. Copying from .env.example...
    copy .env.example .env
    echo [NOTICE] A default '.env' file has been created.
    echo Please open '.env' in VS Code or Notepad and set your GEMINI_API_KEY.
    echo (You can also enter your API key directly inside the Streamlit sidebar).
    echo.
)

:: 6. Launch Streamlit application
echo.
echo ==============================================================================
echo [SUCCESS] Starting Conversational Data Analysis Assistant...
echo Opening in default browser at http://localhost:8501
echo Press Ctrl+C in this terminal to stop the application.
echo ==============================================================================
echo.

python -m streamlit run app.py

pause
