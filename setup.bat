@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  WindowSeat Reflection Removal - Full Setup (Windows)
REM  This script handles everything: Python, CUDA deps, venv, packages.
REM ============================================================

title WindowSeat Reflection Removal - Setup

echo.
echo  ================================================================
echo    WindowSeat Reflection Removal - Automated Setup
echo    ------------------------------------------------
echo    This installer will:
echo      1. Check/install Python 3.11
echo      2. Check NVIDIA GPU and CUDA availability
echo      3. Create a virtual environment
echo      4. Install PyTorch with CUDA support
echo      5. Install all required packages
echo      6. Configure HuggingFace access
echo      7. Download all model weights (~10GB, for offline use)
echo  ================================================================
echo.
pause

REM ============================================================
REM  STEP 1: Check/Install Python
REM ============================================================
echo.
echo [1/6] Checking Python installation...
echo.

set PYTHON_CMD=
set PYTHON_VERSION=

REM Try "python" first
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
    set PYTHON_CMD=python
    goto :python_found
)

REM Try "python3"
python3 --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%v in ('python3 --version 2^>^&1') do set PYTHON_VERSION=%%v
    set PYTHON_CMD=python3
    goto :python_found
)

REM Try common install locations
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    for /f "tokens=2 delims= " %%v in ('"%PYTHON_CMD%" --version 2^>^&1') do set PYTHON_VERSION=%%v
    goto :python_found
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    for /f "tokens=2 delims= " %%v in ('"%PYTHON_CMD%" --version 2^>^&1') do set PYTHON_VERSION=%%v
    goto :python_found
)

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
    for /f "tokens=2 delims= " %%v in ('"%PYTHON_CMD%" --version 2^>^&1') do set PYTHON_VERSION=%%v
    goto :python_found
)

REM Python not found — download and install
echo  Python not found on this system.
echo  Downloading Python 3.12.4 installer...
echo.

set PYTHON_INSTALLER=python-3.12.4-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/3.12.4/%PYTHON_INSTALLER%

REM Use curl (built into Windows 10+) or PowerShell
where curl >nul 2>&1
if not errorlevel 1 (
    curl -L -o "%TEMP%\%PYTHON_INSTALLER%" "%PYTHON_URL%"
) else (
    powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%TEMP%\%PYTHON_INSTALLER%'"
)

if not exist "%TEMP%\%PYTHON_INSTALLER%" (
    echo  ERROR: Failed to download Python installer.
    echo  Please install Python 3.11 manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo  Installing Python 3.12.4 (this may require admin privileges)...
echo  IMPORTANT: Installing with "Add to PATH" enabled.
echo.

"%TEMP%\%PYTHON_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1

if errorlevel 1 (
    echo.
    echo  ERROR: Python installation failed.
    echo  Try running this script as Administrator, or install Python manually:
    echo  https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Refresh PATH
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312\;%LOCALAPPDATA%\Programs\Python\Python312\Scripts\;%PATH%"

REM Verify installation
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Python was installed but is not yet in PATH.
    echo  Please CLOSE this window and run setup.bat again.
    echo  (Windows needs to refresh environment variables)
    pause
    exit /b 1
)

set PYTHON_CMD=python
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
del "%TEMP%\%PYTHON_INSTALLER%" >nul 2>&1

:python_found
echo  [OK] Python %PYTHON_VERSION% found: %PYTHON_CMD%

REM Validate Python version is 3.10+
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo  ERROR: Python 3.10+ required. Found %PYTHON_VERSION%.
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo  ERROR: Python 3.10+ required. Found %PYTHON_VERSION%.
    pause
    exit /b 1
)

REM ============================================================
REM  STEP 2: Check GPU / CUDA
REM ============================================================
echo.
echo [2/6] Checking GPU and CUDA...
echo.

set GPU_NAME=Unknown
set CUDA_AVAILABLE=0

REM Check nvidia-smi
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo  WARNING: nvidia-smi not found.
    echo  NVIDIA drivers may not be installed.
    echo  The application requires an NVIDIA GPU with 12GB+ VRAM.
    echo.
    echo  Download drivers: https://www.nvidia.com/Download/index.aspx
    echo.
    set /p CONTINUE_NO_GPU="  Continue anyway? (y/n): "
    if /i not "!CONTINUE_NO_GPU!"=="y" exit /b 1
    goto :skip_gpu
)

for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2^>nul') do set GPU_NAME=%%g
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2^>nul') do set GPU_VRAM_MB=%%g

set /a GPU_VRAM_GB=%GPU_VRAM_MB% / 1024
echo  [OK] GPU: %GPU_NAME%
echo       VRAM: %GPU_VRAM_GB% GB

if %GPU_VRAM_GB% LSS 8 (
    echo.
    echo  WARNING: Only %GPU_VRAM_GB%GB VRAM detected. Minimum 12GB recommended.
    echo  The app will use 4-bit quantization but may still be slow or OOM.
)

set CUDA_AVAILABLE=1

REM Check CUDA version from nvidia-smi
for /f "tokens=9 delims= " %%c in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%c
if defined CUDA_VERSION (
    echo       CUDA: %CUDA_VERSION%
) else (
    echo       CUDA version: unknown (will install compatible PyTorch)
)

:skip_gpu

REM ============================================================
REM  STEP 3: Create Virtual Environment
REM ============================================================
echo.
echo [3/6] Setting up virtual environment...
echo.

if exist "venv\Scripts\activate.bat" (
    echo  [OK] Virtual environment already exists.
) else (
    echo  Creating virtual environment (venv)...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)

REM Activate venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo  Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ============================================================
REM  STEP 4: Install PyTorch with CUDA
REM ============================================================
echo.
echo [4/6] Installing PyTorch with CUDA support...
echo  (This may take several minutes on first install)
echo.

REM PyTorch 2.8+ requires CUDA 12.4
set TORCH_INDEX=https://download.pytorch.org/whl/cu124

echo  Using PyTorch index: %TORCH_INDEX%
pip install torch torchvision --index-url %TORCH_INDEX%

if errorlevel 1 (
    echo.
    echo  WARNING: PyTorch CUDA install failed. Trying default (may be CPU-only)...
    pip install torch torchvision
)

REM Verify CUDA in PyTorch
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

REM ============================================================
REM  STEP 5: Install All Dependencies
REM ============================================================
echo.
echo [5/6] Installing application dependencies...
echo  (diffusers, transformers, gradio, etc.)
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo  ERROR: Some packages failed to install.
    echo  Check the error messages above. Common fixes:
    echo    - Install Visual C++ Build Tools for bitsandbytes
    echo    - Ensure stable internet connection
    echo.
    echo  Retrying with verbose output...
    pip install -r requirements.txt --verbose
)

REM Install bitsandbytes Windows build specifically
echo.
echo  Installing bitsandbytes (Windows CUDA build)...
pip install bitsandbytes --prefer-binary

REM ============================================================
REM  STEP 6: Configure HuggingFace Token
REM ============================================================
echo.
echo [6/7] HuggingFace Configuration
echo.
echo  The WindowSeat model weights are hosted on HuggingFace.
echo  You need a free HuggingFace account and access token.
echo.
echo  Get your token at: https://huggingface.co/settings/tokens
echo  (Create a token with "Read" access)
echo.

if defined HF_TOKEN (
    echo  [OK] HF_TOKEN is already set in environment.
    goto :download_models
)

set /p HF_INPUT="  Enter your HuggingFace token (or press Enter to skip): "

if not "!HF_INPUT!"=="" (
    REM Save token to HF cache
    python -c "from huggingface_hub import login; login(token='!HF_INPUT!')" 2>nul
    if not errorlevel 1 (
        echo  [OK] Token saved to HuggingFace cache.
    ) else (
        echo  Could not validate token. Setting as environment variable...
    )

    REM Also set as user environment variable for persistence
    setx HF_TOKEN "!HF_INPUT!" >nul 2>&1
    set HF_TOKEN=!HF_INPUT!
    echo  [OK] HF_TOKEN saved as user environment variable.
) else (
    echo.
    echo  WARNING: No token provided. Model download may fail.
    echo  You can set it later with: setx HF_TOKEN hf_your_token_here
    echo.
    set /p SKIP_DL="  Skip model download for now? (y/n): "
    if /i "!SKIP_DL!"=="y" goto :setup_done
)

REM ============================================================
REM  STEP 7: Download Model Weights
REM ============================================================
:download_models
echo.
echo [7/7] Downloading model weights (~10GB)...
echo  This only needs to happen once. After this, the app works offline.
echo.

python download_models.py

if errorlevel 1 (
    echo.
    echo  WARNING: Model download had issues.
    echo  You can retry later by running:
    echo    venv\Scripts\activate.bat
    echo    python download_models.py
    echo.
)

:setup_done
echo.
echo  ================================================================
echo    Setup Complete!
echo  ================================================================
echo.
echo  To run the application:
echo    run.bat
echo.
echo  Or manually:
echo    venv\Scripts\activate.bat
echo    python app.py
echo.
echo  The UI will open at: http://127.0.0.1:7860
echo  First run downloads model weights (~10GB).
echo  ================================================================
echo.
pause
endlocal
