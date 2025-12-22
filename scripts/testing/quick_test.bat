@echo off
REM 快速运行测试并查看报告 - Windows 版本
chcp 65001 >nul 2>&1

echo 🚀 开始运行 APT Model 测试套件...
echo.

REM 1. 基础命令测试
echo ========================================
echo 1️⃣  基础命令测试
echo ========================================
if exist scripts/testing/test_all_commands.py (
    python scripts/testing/test_all_commands.py
    echo.
) else (
    echo ⚠️  scripts/testing/test_all_commands.py 不存在，跳过
    echo.
)

REM 2. 训练后端代码检查
echo ========================================
echo 2️⃣  训练后端代码检查
echo ========================================
if exist tools/check_training_backends.py (
    python tools/check_training_backends.py
    echo.
) else (
    echo ⚠️  tools/check_training_backends.py 不存在，跳过
    echo.
)

REM 3. HLBD系统诊断
echo ========================================
echo 3️⃣  HLBD系统诊断
echo ========================================
if exist tools/diagnose_issues.py (
    python tools/diagnose_issues.py
    echo.
) else (
    echo ⚠️  tools/diagnose_issues.py 不存在，跳过
    echo.
)

REM 4. 生成测试报告
echo ========================================
echo 4️⃣  生成测试报告
echo ========================================
if exist scripts/testing/view_test_report.py (
    python scripts/testing/view_test_report.py
    echo.
) else (
    echo ⚠️  scripts/testing/view_test_report.py 不存在，跳过
    echo.
)

echo.
echo ========================================
echo ✅ 所有测试完成！
echo ========================================
echo.
echo 📂 日志文件位置: test_logs\
echo 💡 你可以将 test_logs\ 目录中的文件发送给开发者进行修复
echo.
echo 🔗 相关文档:
echo    - docs/TRAINING_BACKENDS.md: 训练后端使用指南
echo    - docs/VISUALIZATION_GUIDE.md: 可视化使用指南
echo.
pause
