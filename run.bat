@echo off
setlocal enabledelayedexpansion
title AI Photo Enhancer

echo.
echo  AI Photo Enhancer
echo  ==============================
echo.

REM Check venv exists
if not exist "venv\Scripts\activate.bat" (
    echo  ERROR: Virtual environment not found.
    echo  Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check if key packages are installed
python -c "import torch, gradio, diffusers" >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Required packages not installed.
    echo  Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

REM Show GPU status
echo  GPU Check:
python -c "import torch; print(f'    CUDA available: {torch.cuda.is_available()}'); print(f'    Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
echo.

REM Check HF_TOKEN
if not defined HF_TOKEN (
    python -c "from huggingface_hub import HfFolder; t=HfFolder.get_token(); exit(0 if t else 1)" >nul 2>&1
    if errorlevel 1 (
        echo  WARNING: No HuggingFace token found.
        echo  Model download may fail. Set with: setx HF_TOKEN hf_your_token
        echo.
    )
)

echo  Starting application...
echo  UI will open at: http://127.0.0.1:7860
echo  Press Ctrl+C to stop.
echo.

python app.py

if errorlevel 1 (
    echo.
    echo  Application exited with an error.
    echo  Check the messages above for details.
)

pause
endlocal
