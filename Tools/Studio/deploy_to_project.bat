@echo off
:: ============================================================================
:: deploy_to_project.bat
:: EdgeSense ST Edge AI Deployment & Firmware Build Pipeline
::
:: EdgeSense: STM32 Edge AI Texture and Gesture Classifier
::
:: Copyright (c) 2026 Dharagesh and Circuit Digest
:: https://github.com/Circuit-Digest/EdgeSense
:: Licensed under GNU General Public License v3.0
::
:: Usage:
::   deploy_to_project.bat          (Deploys all trained models found in Models/)
::   deploy_to_project.bat all      (Deploys all trained models found in Models/)
::   deploy_to_project.bat gesture  (Deploys 3D Gesture 2D-CNN)
::   deploy_to_project.bat posture  (Deploys Hand Posture 2D-CNN)
::   deploy_to_project.bat surface  (Deploys Surface 1D-CNN)
:: ============================================================================
setlocal enabledelayedexpansion

set "MODE=%~1"
if "%MODE%"=="" set "MODE=all"

echo ======================================================================
echo EdgeSense - Neural Network Deployment to Project
echo Target Mode: %MODE%
echo ======================================================================
echo.

set "ROOT_DIR=%~dp0..\.."
if not exist "%ROOT_DIR%\Models" (
    if exist "%~dp0..\..\CODE\EdgeSense\Models" (
        set "ROOT_DIR=%~dp0..\..\CODE\EdgeSense"
    )
)

set "OUT_DIR=%ROOT_DIR%\X-CUBE-AI\App"
set "MODELS_DIR=%ROOT_DIR%\Models"

call "%~dp0..\find_st_tools.bat"

if not defined STEDGEAI (
    echo [ERROR] ST Edge AI Core [stedgeai.exe] not found!
    echo Please install STM32Cube.AI Studio or X-CUBE-AI pack, or add stedgeai to your PATH.
    exit /b 1
)

set "CONVERTED_COUNT=0"
set "DEPLOYED_MODELS="

:: ----------------------------------------------------------------------------
:: 1. Deploy Gesture Model
:: ----------------------------------------------------------------------------
if "%MODE%"=="gesture" goto :do_gesture
if "%MODE%"=="all" goto :do_gesture
goto :skip_gesture

:do_gesture
set "MODEL_GESTURE=%MODELS_DIR%\gesture_nn_model.onnx"
if not exist "%MODEL_GESTURE%" (
    if "%MODE%"=="gesture" (
        echo [ERROR] Gesture model not found: %MODEL_GESTURE%
        echo Please train a gesture model in the EdgeSense Studio first.
        exit /b 1
    )
    goto :skip_gesture
)
echo [Converting] 3D Gesture Recognition ONNX model with ST Edge AI Core v2 [st-ai]...
"%STEDGEAI%" generate --target stm32h5 --name gesture_nn -m "%MODEL_GESTURE%" --c-api st-ai --compression none --no-workspace --output "%OUT_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] ST Edge AI code generation failed for gesture_nn.
    exit /b 1
)
python -c "p=r'%OUT_DIR%\gesture_nn.c'; t=open(p,'r').read().replace('\nvoid forward_lite_', '\nstatic void forward_lite_'); open(p,'w').write(t)"
set /a CONVERTED_COUNT+=1
set "DEPLOYED_MODELS=!DEPLOYED_MODELS! [Gesture NN]"

:skip_gesture

:: ----------------------------------------------------------------------------
:: 2. Deploy Surface Model
:: ----------------------------------------------------------------------------
if "%MODE%"=="surface" goto :do_surface
if "%MODE%"=="all" goto :do_surface
goto :skip_surface

:do_surface
set "MODEL_SURFACE=%MODELS_DIR%\surface_nn_model.onnx"
if not exist "%MODEL_SURFACE%" (
    if "%MODE%"=="surface" (
        echo [ERROR] Surface model not found: %MODEL_SURFACE%
        echo Please train a surface model in the EdgeSense Studio first.
        exit /b 1
    )
    goto :skip_surface
)
echo [Converting] Surface 1D-CNN ONNX model with ST Edge AI Core v2 [st-ai]...
"%STEDGEAI%" generate --target stm32h5 --name surface_nn -m "%MODEL_SURFACE%" --c-api st-ai --compression none --no-workspace --output "%OUT_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] ST Edge AI code generation failed for surface_nn.
    exit /b 1
)
python -c "p=r'%OUT_DIR%\surface_nn.c'; t=open(p,'r').read().replace('\nvoid forward_lite_', '\nstatic void forward_lite_'); open(p,'w').write(t)"
set /a CONVERTED_COUNT+=1
set "DEPLOYED_MODELS=!DEPLOYED_MODELS! [Surface 1D-CNN]"

:skip_surface

:: ----------------------------------------------------------------------------
:: 3. Deploy Posture Model
:: ----------------------------------------------------------------------------
if "%MODE%"=="posture" goto :do_posture
if "%MODE%"=="all" goto :do_posture
goto :skip_posture

:do_posture
set "MODEL_POSTURE=%MODELS_DIR%\posture_nn_model.onnx"
if not exist "%MODEL_POSTURE%" (
    if "%MODE%"=="posture" (
        echo [ERROR] Posture model not found: %MODEL_POSTURE%
        echo Please train a posture model in the EdgeSense Studio first.
        exit /b 1
    )
    goto :skip_posture
)
echo [Converting] Hand Posture 2D-CNN ONNX model with ST Edge AI Core v2 [st-ai]...
"%STEDGEAI%" generate --target stm32h5 --name posture_nn -m "%MODEL_POSTURE%" --c-api st-ai --compression none --no-workspace --output "%OUT_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] ST Edge AI code generation failed for posture_nn.
    exit /b 1
)
python -c "p=r'%OUT_DIR%\posture_nn.c'; t=open(p,'r').read().replace('\nvoid forward_lite_', '\nstatic void forward_lite_'); open(p,'w').write(t)"
set /a CONVERTED_COUNT+=1
set "DEPLOYED_MODELS=!DEPLOYED_MODELS! [Posture 2D-CNN]"

:skip_posture

:: ----------------------------------------------------------------------------
:: 4. Verify At Least One Model Was Converted
:: ----------------------------------------------------------------------------
if %CONVERTED_COUNT% equ 0 (
    echo [WARNING] No trained ONNX models found in: %MODELS_DIR%
    echo Please train a Gesture, Surface, or Posture model in the EdgeSense Studio first.
    exit /b 1
)

:: ----------------------------------------------------------------------------
:: 5. Compile Firmware Binary
:: ----------------------------------------------------------------------------
echo.
if defined MAKE_EXE (
    echo [Compiling] Building EdgeSense firmware with STM32 GNU toolchain...
    "%MAKE_EXE%" -C "%ROOT_DIR%\Debug" all
    if %errorlevel% equ 0 (
        arm-none-eabi-objcopy -O binary "%ROOT_DIR%\Debug\EdgeSense.elf" "%ROOT_DIR%\Debug\EdgeSense.bin" 2>nul
        arm-none-eabi-objcopy -O ihex "%ROOT_DIR%\Debug\EdgeSense.elf" "%ROOT_DIR%\Debug\EdgeSense.hex" 2>nul
    ) else (
        echo [WARNING] Build returned error. You can compile directly in STM32CubeIDE.
    )
) else (
    echo [NOTE] GNU make not detected in PATH.
    echo C Sources generated. Open STM32CubeIDE and click 'Build' to compile the firmware.
)

echo.
echo ======================================================================
echo SUCCESS: Model(s) deployed to EdgeSense project!
echo Deployed Models :%DEPLOYED_MODELS%
echo C Sources updated: %OUT_DIR%
if exist "%ROOT_DIR%\Debug\EdgeSense.bin" echo Firmware binary ready: %ROOT_DIR%\Debug\EdgeSense.bin
echo.
echo Flash to your device using your preferred method:
echo   - USB-C DFU : Tools\Flashing\flash_firmware_usb.bat
echo   - ST-LINK   : Tools\Flashing\flash_firmware.bat
echo   - Or open STM32CubeIDE and click Run / Debug
echo ======================================================================
exit /b 0
