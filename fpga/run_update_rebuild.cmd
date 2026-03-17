@echo off
cd /d "%~dp0"

set VIVADO_CMD=

where vivado >nul 2>nul
if %errorlevel%==0 set VIVADO_CMD=vivado

if not defined VIVADO_CMD if defined XILINX_VIVADO if exist "%XILINX_VIVADO%\bin\vivado.bat" set VIVADO_CMD="%XILINX_VIVADO%\bin\vivado.bat"
if not defined VIVADO_CMD if exist "C:\Xilinx\Vivado\2024.1\bin\vivado.bat" set VIVADO_CMD="C:\Xilinx\Vivado\2024.1\bin\vivado.bat"
if not defined VIVADO_CMD if exist "C:\AMD\Vivado\2024.1\bin\vivado.bat" set VIVADO_CMD="C:\AMD\Vivado\2024.1\bin\vivado.bat"

if not defined VIVADO_CMD (
    echo Vivado not found.
    echo Add Vivado to PATH or set XILINX_VIVADO, or install in a standard location.
    pause
    exit /b 1
)

call %VIVADO_CMD% -mode batch -source update_and_rebuild.tcl
pause
