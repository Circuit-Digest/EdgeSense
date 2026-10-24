@echo off
:: ============================================================================
:: import_cubeai_zip.bat
:: 1-Click Importer for STM32Cube.AI Studio Generated Zip Code
::
:: EdgeSense: STM32 Edge AI Texture and Gesture Classifier
:: Copyright (c) 2026 Dharagesh and Circuit Digest
:: ============================================================================
setlocal

echo ======================================================================
echo EdgeSense - STM32Cube.AI Studio C-Model Auto-Importer
echo ======================================================================
echo.

python "%~dp0import_cubeai_zip.py" %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Import failed. Please ensure you clicked "Download your C model" in Studio.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo SUCCESS: Model files imported to CODE\EdgeSense\X-CUBE-AI\App\
echo Ready to build in STM32CubeIDE!
echo ======================================================================
timeout /t 5
