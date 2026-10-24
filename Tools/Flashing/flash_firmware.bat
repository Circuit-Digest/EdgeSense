@echo off
:: ============================================================================
:: flash_firmware.bat
:: Firmware Flashing Tool for EdgeSense (STM32H533 via ST-LINK)
::
:: EdgeSense: STM32 Edge AI Texture and Gesture Classifier
::
:: Copyright (c) 2026 Dharagesh and Circuit Digest
:: https://github.com/Circuit-Digest/EdgeSense
:: Licensed under GNU General Public License v3.0
:: ============================================================================
setlocal
echo ======================================================================
echo EdgeSense - STM32H533 Firmware Flashing Tool (ST-LINK)
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

echo Flashing %BIN_FILE% to STM32H533 @ 0x08000000...
"%PROG_CLI%" -c port=SWD mode=UR -w "%BIN_FILE%" 0x08000000 -v -rst
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Flashing failed!
    echo Ensure the ST-LINK is connected to the J3 header:
    echo   Pin 1: 3.3V, Pin 2: GND, Pin 3: NRST [required for mode=UR], Pin 4: SWDIO, Pin 5: SWCLK
    echo Alternatively, you can flash via USB-C using flash_firmware_usb.bat.
    exit /b 1
)

echo.
echo ======================================================================
echo SUCCESS: Firmware flashed and verified! STM32H533 is running.
echo ======================================================================
