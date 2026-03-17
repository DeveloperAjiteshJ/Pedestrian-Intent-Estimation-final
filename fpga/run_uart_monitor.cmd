@echo off
cd /d "%~dp0.."
python fpga_uart_monitor.py --port COM3
pause
