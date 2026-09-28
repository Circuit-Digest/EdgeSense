@echo off
:: ============================================================================
:: flash_firmware.bat
:: Firmware Flashing Tool for EdgeSense (STM32H533)
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
echo EdgeSense - STM32H533 Firmware Flashing Tool
echo ======================================================================
echo.

set "PROG_CLI=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin\STM32_Programmer_CLI.exe"
set "ROOT_DIR=%~dp0..\.."
set "BIN_FILE=%ROOT_DIR%\Debug\EdgeSense.bin"

if not exist "%BIN_FILE%" (
    echo [ERROR] %BIN_FILE% not found. Please compile the firmware first.
    exit /b 1
)

echo Flashing %BIN_FILE% to STM32H533 @ 0x08000000...
"%PROG_CLI%" -c port=SWD mode=UR -w "%BIN_FILE%" 0x08000000 -v -rst
if %errorlevel% neq 0 (
    echo [ERROR] Flashing failed!
    exit /b 1
)

echo.
echo ======================================================================
echo SUCCESS: Firmware flashed and verified! STM32H533 is running.
echo ======================================================================
