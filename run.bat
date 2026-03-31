@echo off
echo ================================================================================
echo                          ML-IDS - Quick Launcher
echo ================================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Setting up for the first time...
    echo.
    echo [1/5] Checking for Python 3.11...
    py -3.11 --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Python 3.11 not found!
        echo Please install from: https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
    
    echo [2/5] Creating virtual environment...
    py -3.11 -m venv venv
    
    echo [3/5] Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo [4/5] Upgrading pip...
    python -m pip install --upgrade pip --quiet
    
    echo [5/5] Installing requirements (this may take 5-10 minutes)...
    pip install -r requirements.txt --quiet
    
    echo.
    echo ================================================================================
    echo                     Setup Complete! Starting app...
    echo ================================================================================
    echo.
) else (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if app exists
if not exist "app\streamlit_app.py" (
    echo ERROR: streamlit_app.py not found!
    pause
    exit /b 1
)

echo.
echo Starting ML-IDS Streamlit App...
echo.
echo App will open at: http://localhost:8501
echo Press Ctrl+C to stop
echo.
echo ================================================================================
echo.

streamlit run app\streamlit_app.py

pause
