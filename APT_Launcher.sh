#!/bin/bash
# APT Transformer启动器 - Linux/macOS Shell版本
# 双击此文件即可启动GUI启动器

echo "=========================================="
echo "  APT Transformer 启动器"
echo "=========================================="
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python 3，请先安装Python 3.8+"
    echo
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS安装方法:"
        echo "  brew install python3"
        echo "或下载: https://www.python.org/downloads/"
    else
        echo "Ubuntu/Debian安装方法:"
        echo "  sudo apt-get install python3 python3-tk"
        echo
        echo "Fedora/CentOS安装方法:"
        echo "  sudo dnf install python3 python3-tkinter"
    fi
    read -p "按任意键退出..."
    exit 1
fi

echo "[检测] Python已安装"
python3 --version
echo

# 检查tkinter
echo "[检查] 检查tkinter..."
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "[警告] tkinter未安装"
    echo
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS通常自带tkinter，如果启动失败，请重新安装Python"
    else
        echo "Ubuntu/Debian安装方法:"
        echo "  sudo apt-get install python3-tk"
        echo
        echo "Fedora/CentOS安装方法:"
        echo "  sudo dnf install python3-tkinter"
    fi
    echo
    read -p "是否继续尝试启动？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "[检测] tkinter已安装"
fi

echo
echo "[启动] 正在启动APT GUI启动器..."
echo

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

# 启动GUI启动器
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    python3 apt_launcher.pyw &
else
    # Linux
    python3 apt_launcher.pyw &
fi

# 检查启动是否成功
if [ $? -eq 0 ]; then
    echo "[成功] GUI启动器已启动"
    echo
    exit 0
else
    echo
    echo "[错误] 启动失败，请检查:"
    echo "  1. 是否在项目根目录"
    echo "  2. apt_launcher.pyw文件是否存在"
    echo "  3. Python环境是否正确配置"
    echo "  4. tkinter是否正确安装"
    echo
    read -p "按任意键退出..."
    exit 1
fi
