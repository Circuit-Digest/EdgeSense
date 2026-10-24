@echo off
:: ============================================================================
:: launch_cubeai_studio.bat
:: Launches STMicroelectronics STM32Cube.AI Studio (STEDGEAI-CUBEAI)
::
:: EdgeSense: STM32 Edge AI Texture and Gesture Classifier
:: Copyright (c) 2026 Dharagesh and Circuit Digest
:: ============================================================================

set "STUDIO_EXE=C:\ST\STM32Cube-AI-Studio\STM32Cube-AI-Studio.exe"

if not exist "%STUDIO_EXE%" (
    echo [ERROR] STM32Cube-AI Studio executable not found at:
    echo         %STUDIO_EXE%
    echo Please install STEDGEAI-CUBEAI from st.com.
    pause
    exit /b 1
)

echo ======================================================================
echo Launching STM32Cube.AI Studio (STEDGEAI-CUBEAI)...
echo ======================================================================
echo Model files ready for import:
echo   - Gesture: CODE\EdgeSense\Models\gesture_nn_model.onnx
echo   - Surface: CODE\EdgeSense\Models\surface_nn_model.onnx
echo   - Posture: CODE\EdgeSense\Models\posture_nn_model.onnx
echo Target MCU: STM32H533CEUx
echo Output Directory: CODE\EdgeSense\X-CUBE-AI\App
echo ======================================================================
echo.

start "" "%STUDIO_EXE%"
exit /b 0
