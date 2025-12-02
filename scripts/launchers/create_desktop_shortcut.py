#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
桌面快捷方式生成器
为APT Launcher创建桌面快捷方式（支持Windows、Linux、macOS）
"""

import os
import sys
import platform
from pathlib import Path


def get_desktop_path():
    """获取桌面路径"""
    system = platform.system()

    if system == "Windows":
        desktop = Path.home() / "Desktop"
        # 也可能是中文"桌面"
        if not desktop.exists():
            desktop = Path.home() / "桌面"
        return desktop

    elif system == "Darwin":  # macOS
        return Path.home() / "Desktop"

    else:  # Linux
        # Try XDG user dirs first
        try:
            import subprocess
            result = subprocess.run(
                ["xdg-user-dir", "DESKTOP"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass

        # Fallback to ~/Desktop
        return Path.home() / "Desktop"


def create_windows_shortcut(project_root, desktop_path):
    """创建Windows快捷方式"""
    try:
        import win32com.client
    except ImportError:
        print("[错误] 缺少pywin32模块")
        print("请运行: pip install pywin32")
        return False

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut_path = desktop_path / "APT Launcher.lnk"

    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.TargetPath = str(project_root / "APT_Launcher.bat")
    shortcut.WorkingDirectory = str(project_root)
    shortcut.IconLocation = str(project_root / "APT_Launcher.bat")
    shortcut.Description = "APT Transformer 启动器"
    shortcut.save()

    print(f"[成功] Windows快捷方式已创建: {shortcut_path}")
    return True


def create_linux_shortcut(project_root, desktop_path):
    """创建Linux桌面文件"""
    desktop_file = desktop_path / "apt-launcher.desktop"

    content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=APT Launcher
Comment=APT Transformer启动器
Exec={project_root}/APT_Launcher.sh
Icon=utilities-terminal
Terminal=false
Categories=Development;Science;
Path={project_root}
"""

    desktop_file.write_text(content)

    # 设置可执行权限
    os.chmod(desktop_file, 0o755)

    print(f"[成功] Linux桌面文件已创建: {desktop_file}")
    print("[提示] 如果无法双击启动，请右键 -> 属性 -> 勾选'允许作为程序执行'")
    return True


def create_macos_shortcut(project_root, desktop_path):
    """创建macOS应用程序包"""
    app_path = desktop_path / "APT Launcher.app"
    contents_dir = app_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"

    # 创建目录结构
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    # 创建Info.plist
    info_plist = contents_dir / "Info.plist"
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleName</key>
    <string>APT Launcher</string>
    <key>CFBundleIdentifier</key>
    <string>com.apt.launcher</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
"""
    info_plist.write_text(plist_content)

    # 创建启动脚本
    launcher_script = macos_dir / "launcher"
    script_content = f"""#!/bin/bash
cd "{project_root}"
exec python3 apt_launcher.pyw
"""
    launcher_script.write_text(script_content)
    os.chmod(launcher_script, 0o755)

    print(f"[成功] macOS应用程序已创建: {app_path}")
    print("[提示] 首次运行可能需要在'系统偏好设置 -> 安全性与隐私'中允许")
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("  APT Transformer 桌面快捷方式生成器")
    print("=" * 60)
    print()

    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    print(f"[检测] 项目路径: {project_root}")

    # 获取桌面路径
    desktop_path = get_desktop_path()
    print(f"[检测] 桌面路径: {desktop_path}")

    if not desktop_path.exists():
        print(f"[错误] 桌面路径不存在: {desktop_path}")
        return 1

    print()

    # 检测操作系统并创建快捷方式
    system = platform.system()
    print(f"[检测] 操作系统: {system}")
    print()

    try:
        if system == "Windows":
            success = create_windows_shortcut(project_root, desktop_path)

        elif system == "Darwin":  # macOS
            success = create_macos_shortcut(project_root, desktop_path)

        elif system == "Linux":
            success = create_linux_shortcut(project_root, desktop_path)

        else:
            print(f"[错误] 不支持的操作系统: {system}")
            return 1

        if success:
            print()
            print("=" * 60)
            print("✓ 快捷方式创建成功！")
            print("=" * 60)
            print()
            print("现在您可以在桌面上双击'APT Launcher'图标启动程序")
            print()
            return 0
        else:
            return 1

    except Exception as e:
        print(f"[错误] 创建快捷方式失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()

    if platform.system() == "Windows":
        input("\n按Enter键退出...")

    sys.exit(exit_code)
