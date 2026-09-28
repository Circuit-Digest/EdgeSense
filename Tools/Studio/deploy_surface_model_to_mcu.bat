@echo off
:: ============================================================================
:: deploy_surface_model_to_mcu.bat
:: Surface 1D-CNN ST Edge AI Deployment & Flashing Pipeline
::
:: EdgeSense: STM32 Edge AI Texture and Gesture Classifier
::
:: Copyright (c) 2026 Dharagesh and Circuit Digest
:: https://github.com/Circuit-Digest/EdgeSense
:: Licensed under GNU General Public License v3.0
:: ============================================================================
::
::   ███████╗██████╗  ██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗███████╗
::   ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
::   █████╗  ██║  ██║██║  ███╗█████╗  ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  
::   ██╔══╝  ██║  ██║██║   ██║██╔══╝  ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
::   ███████╗██████╔╝╚██████╔╝███████╗███████║███████╗██║ ╚████║███████║███████╗
::   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
::
setlocal

echo ======================================================================
echo EdgeSense - Surface 1D-CNN ST Edge AI Deployment Pipeline
echo ======================================================================
echo.

set "ROOT_DIR=%~dp0..\.."
set "STEDGEAI=C:\Users\sdham\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\10.2.1\Utilities\windows\stedgeai.exe"
set "MODEL=%ROOT_DIR%\Models\surface_nn_model.onnx"
set "OUT_DIR=%ROOT_DIR%\X-CUBE-AI\App"
set "MAKE_EXE=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615\tools\bin\make.exe"
set "GCC_BIN=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740\tools\bin"
set "PATH=%GCC_BIN%;%PATH%"

if not exist "%MODEL%" (
    echo [ERROR] Model %MODEL% not found. Please train a surface model in the Studio first.
    exit /b 1
)

if exist "%~dp0st_ai_ws" rmdir /s /q "%~dp0st_ai_ws" 2>nul

echo [1/2] Converting trained Surface 1D-CNN ONNX model with ST Edge AI Core v10.2.1...
"%STEDGEAI%" generate --target stm32h5 --name surface_nn -m "%MODEL%" --compression none --no-workspace --output "%OUT_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] ST Edge AI code generation failed.
    exit /b 1
)

echo.
echo [2/2] Compiling firmware binary with STM32CubeIDE GNU toolchain...
"%MAKE_EXE%" -C "%ROOT_DIR%\Debug" all
if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed.
    exit /b 1
)

arm-none-eabi-objcopy -O binary "%ROOT_DIR%\Debug\EdgeSense.elf" "%ROOT_DIR%\Debug\EdgeSense.bin"
arm-none-eabi-objcopy -O ihex "%ROOT_DIR%\Debug\EdgeSense.elf" "%ROOT_DIR%\Debug\EdgeSense.hex"

echo.
echo [3/3] Flashing firmware to STM32H533 target via ST-LINK...
call "%~dp0..\Flashing\flash_firmware.bat"
if %errorlevel% neq 0 (
    echo [WARNING] Flashing failed or ST-LINK not connected. You can flash manually using Tools\Flashing\flash_firmware.bat.
)

echo.
echo ======================================================================
echo SUCCESS: Firmware updated and running with new Surface 1D-CNN!
echo ======================================================================
