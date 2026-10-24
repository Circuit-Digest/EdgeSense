@echo off
:: ============================================================================
:: find_st_tools.bat
:: Dynamic STMicroelectronics Toolchain Auto-Discovery
:: Locates stedgeai, arm-none-eabi-gcc, make, and STM32CubeProgrammer
:: without hardcoded user directories.
:: ============================================================================

:: 1. Discover stedgeai (ST Edge AI Core / X-CUBE-AI)
set "STEDGEAI="
where stedgeai.exe >nul 2>&1
if %errorlevel% equ 0 (
    set "STEDGEAI=stedgeai.exe"
)
if not defined STEDGEAI (
    if defined USERPROFILE (
        for /d %%D in ("%USERPROFILE%\STM32Cube\Repository\Packs\STMicroelectronics\X-CUBE-AI\*") do (
            if exist "%%D\Utilities\windows\stedgeai.exe" set "STEDGEAI=%%D\Utilities\windows\stedgeai.exe"
        )
    )
)
if not defined STEDGEAI (
    for /d %%D in ("C:\ST\STEdgeAI\*") do (
        if exist "%%D\Utilities\windows\stedgeai.exe" set "STEDGEAI=%%D\Utilities\windows\stedgeai.exe"
        if exist "%%D\stedgeai.exe" set "STEDGEAI=%%D\stedgeai.exe"
    )
)
if not defined STEDGEAI (
    if defined LOCALAPPDATA (
        for /d %%D in ("%LOCALAPPDATA%\Programs\STMicroelectronics\STEdgeAI\*") do (
            if exist "%%D\stedgeai.exe" set "STEDGEAI=%%D\stedgeai.exe"
        )
    )
)
if not defined STEDGEAI (
    if defined ProgramFiles (
        for /d %%D in ("%ProgramFiles%\STMicroelectronics\STEdgeAI\*") do (
            if exist "%%D\stedgeai.exe" set "STEDGEAI=%%D\stedgeai.exe"
        )
    )
)

:: 2. Discover STM32CubeIDE GNU GCC & Make
set "MAKE_EXE="
where make.exe >nul 2>&1
if %errorlevel% equ 0 (
    set "MAKE_EXE=make.exe"
)
if not defined MAKE_EXE (
    for /d %%D in ("C:\ST\STM32CubeIDE*") do (
        for /d %%P in ("%%D\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.make*") do (
            if exist "%%P\tools\bin\make.exe" set "MAKE_EXE=%%P\tools\bin\make.exe"
        )
    )
)

set "GCC_BIN="
where arm-none-eabi-gcc.exe >nul 2>&1
if %errorlevel% equ 0 (
    set "GCC_BIN=FOUND"
)
if not defined GCC_BIN (
    for /d %%D in ("C:\ST\STM32CubeIDE*") do (
        for /d %%P in ("%%D\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32*") do (
            if exist "%%P\tools\bin\arm-none-eabi-gcc.exe" set "GCC_BIN=%%P\tools\bin"
        )
    )
)
if defined GCC_BIN (
    if not "%GCC_BIN%"=="FOUND" (
        set "PATH=%GCC_BIN%;%PATH%"
    )
)

:: 3. Discover STM32CubeProgrammer CLI
set "PROG_CLI="
where STM32_Programmer_CLI.exe >nul 2>&1
if %errorlevel% equ 0 (
    set "PROG_CLI=STM32_Programmer_CLI.exe"
)
if not defined PROG_CLI (
    for /d %%D in ("C:\ST\STM32CubeIDE*") do (
        for /d %%P in ("%%D\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer*") do (
            if exist "%%P\tools\bin\STM32_Programmer_CLI.exe" set "PROG_CLI=%%P\tools\bin\STM32_Programmer_CLI.exe"
        )
    )
)
if not defined PROG_CLI (
    if exist "%ProgramFiles%\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe" (
        set "PROG_CLI=%ProgramFiles%\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"
    )
)

exit /b 0
