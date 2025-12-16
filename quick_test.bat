@echo off
REM 快速运行测试并查看报告 - Windows 版本
chcp 65001 >nul 2>&1

echo 🚀 开始运行 APT Model 命令测试...
echo.

REM 运行测试
python test_all_commands.py

REM 查看报告
echo.
echo 📊 生成报告...
python view_test_report.py

echo.
echo ✅ 测试完成！
echo.
echo 日志文件位置: test_logs\
echo 你可以将 test_logs\ 目录中的文件发送给开发者进行修复
pause
