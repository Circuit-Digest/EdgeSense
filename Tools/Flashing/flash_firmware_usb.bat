@echo off
:: ============================================================================
:: flash_firmware_usb.bat
:: Firmware Flashing Tool for EdgeSense (STM32H533 via USB-C DFU)
::
:: EdgeSense: STM32 Edge AI Texture and Gesture Classifier
::
:: Copyright (c) 2026 Dharagesh and Circuit Digest
:: https://github.com/Circuit-Digest/EdgeSense
:: Licensed under GNU General Public License v3.0
:: ============================================================================
setlocal
echo ======================================================================
echo EdgeSense - STM32H533 USB DFU Flashing Tool
echo ======================================================================
echo.

set "ROOT_DIR=%~dp0..\.."
set "BIN_FILE=%ROOT_DIR%\Debug\EdgeSense.bin"

call "%~dp0..\find_st_tools.bat"

if not defined PROG_CLI (
    echo [ERROR] STM32_Programmer_CLI not found!
    echo Please install STM32CubeProgrammer or STM32CubeIDE, or add STM32_Programmer_CLI to PATH.
    exit /b 1
)

if not exist "%BIN_FILE%" (
    echo [ERROR] %BIN_FILE% not found. Please compile the firmware first.
    exit /b 1
)

echo Flashing %BIN_FILE% to STM32H533 via USB DFU @ 0x08000000...
"%PROG_CLI%" -c port=usb1 -w "%BIN_FILE%" 0x08000000 -v -rst
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] USB DFU Flashing failed!
    echo Ensure the board is connected via USB-C and in DFU mode:
    echo   1. Press and HOLD SW2 [BOOT0]
    echo   2. Press and RELEASE SW1 [RESET]
    echo   3. Release SW2 [BOOT0]
    exit /b 1
)

echo.
echo ======================================================================
echo SUCCESS: Firmware flashed via USB-C! STM32H533 is running.
echo ======================================================================
