@echo off
setlocal enabledelayedexpansion
title Food Delivery Delay Prediction - DSML System

:: Navigate to project directory
cd /d "%~dp0"

echo ===============================================================================
echo            FOOD DELIVERY DELAY PREDICTION - DSML SYSTEM
echo            Author: Jaival Patel ^| B.Tech CSE (23000300)
echo ===============================================================================
echo.

:: 1. Detect Python command
set "PYTHON_CMD="
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
) else (
    py --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_CMD=py"
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/ and check
    echo the option "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [*] Python detected: 
%PYTHON_CMD% --version
echo.

:: 2. Check and handle optional arguments
if /i "%~1"=="--train" goto do_train
if /i "%~1"=="--test" goto do_test
if /i "%~1"=="--predict" goto do_predict
goto check_dependencies

:do_train
echo [*] Mode: Training full pipeline...
%PYTHON_CMD% run_pipeline.py --mode all
echo.
pause
exit /b 0

:do_test
echo [*] Mode: Running test suite...
%PYTHON_CMD% -m unittest discover -s tests -p "test_*.py" -v
echo.
pause
exit /b 0

:do_predict
echo [*] Mode: Slide 9 Single Prediction...
%PYTHON_CMD% run_pipeline.py --mode predict-sample
echo.
pause
exit /b 0

:check_dependencies
:: 3. Verify essential Python packages are available
echo [*] Checking dependencies...
%PYTHON_CMD% -c "import flask, sklearn, pandas, joblib, matplotlib, seaborn" >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [!] Missing dependencies detected. Installing required packages...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install requirements. Please check your internet connection.
        pause
        exit /b 1
    )
    echo [*] All dependencies installed successfully.
) else (
    echo [*] Dependencies verified.
)
echo.

:: 4. Verify trained model exists
if not exist "models\random_forest_regressor.joblib" (
    echo [!] Trained model artifact not found.
    echo [*] Training Random Forest Regressor on dataset [this may take ~1-2 minutes]...
    %PYTHON_CMD% run_pipeline.py --mode all
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Model training failed!
        pause
        exit /b 1
    )
    echo [*] Model training completed successfully!
    echo.
)

:: 5. Launch web application and auto-open browser
echo ===============================================================================
echo  [*] Starting Web Application on http://127.0.0.1:5000
echo  [*] Opening interactive dashboard in your default browser...
echo  [*] Press Ctrl+C in this console window to stop the server at any time.
echo ===============================================================================
echo.

:: Launch background browser opener (waits 2 seconds for Flask to bind port)
start /b cmd /c "ping 127.0.0.1 -n 3 >nul & start http://127.0.0.1:5000"

:: Start Flask server in foreground
%PYTHON_CMD% app\app.py

echo.
echo [*] Web server has been stopped.
pause

