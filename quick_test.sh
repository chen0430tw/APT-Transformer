#!/bin/bash
# 快速运行测试并查看报告 - Linux/Mac 版本

echo "🚀 开始运行 APT Model 测试套件..."
echo ""

# 1. 基础命令测试
echo "========================================"
echo "1️⃣  基础命令测试"
echo "========================================"
if [ -f "test_all_commands.py" ]; then
    python test_all_commands.py
    echo ""
else
    echo "⚠️  test_all_commands.py 不存在，跳过"
    echo ""
fi

# 2. 训练后端代码检查
echo "========================================"
echo "2️⃣  训练后端代码检查"
echo "========================================"
if [ -f "check_training_backends.py" ]; then
    python check_training_backends.py
    echo ""
else
    echo "⚠️  check_training_backends.py 不存在，跳过"
    echo ""
fi

# 3. HLBD系统诊断
echo "========================================"
echo "3️⃣  HLBD系统诊断"
echo "========================================"
if [ -f "diagnose_issues.py" ]; then
    python diagnose_issues.py
    echo ""
else
    echo "⚠️  diagnose_issues.py 不存在，跳过"
    echo ""
fi

# 4. 生成测试报告
echo "========================================"
echo "4️⃣  生成测试报告"
echo "========================================"
if [ -f "view_test_report.py" ]; then
    python view_test_report.py
    echo ""
else
    echo "⚠️  view_test_report.py 不存在，跳过"
    echo ""
fi

echo ""
echo "========================================"
echo "✅ 所有测试完成！"
echo "========================================"
echo ""
echo "📂 日志文件位置: test_logs/"
echo "💡 你可以将 test_logs/ 目录中的文件发送给开发者进行修复"
echo ""
echo "🔗 相关文档:"
echo "   - TRAINING_BACKENDS.md: 训练后端使用指南"
echo "   - VISUALIZATION_GUIDE.md: 可视化使用指南"
echo ""
