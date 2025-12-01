@echo off
REM APT Transformer启动器 - Windows批处理版本
REM 双击此文件即可启动GUI启动器

title APT Transformer Launcher

echo ==========================================
echo   APT Transformer 启动器
echo ==========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    echo.
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [检测] Python已安装
python --version
echo.

REM 检查必要的包
echo [检查] 检查依赖包...
python -c "import tkinter" 2>nul
if errorlevel 1 (
    echo [警告] tkinter未安装，但通常随Python自带
    echo 如果启动失败，请重新安装Python并确保勾选tk/tcl组件
    echo.
)

REM 启动GUI启动器
echo [启动] 正在启动APT GUI启动器...
echo.

pythonw apt_launcher.pyw

REM 如果pythonw不可用，使用python
if errorlevel 1 (
    python apt_launcher.pyw
)

if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查:
    echo   1. 是否在项目根目录
    echo   2. apt_launcher.pyw文件是否存在
    echo   3. Python环境是否正确配置
    pause
    exit /b 1
)

exit /b 0
