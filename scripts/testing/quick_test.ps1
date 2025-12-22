# 快速运行测试并查看报告 - PowerShell 版本

Write-Host "🚀 开始运行 APT Model 命令测试..." -ForegroundColor Green
Write-Host ""

# 运行测试
python scripts/testing/test_all_commands.py

# 查看报告
Write-Host ""
Write-Host "📊 生成报告..." -ForegroundColor Cyan
python scripts/testing/view_test_report.py

Write-Host ""
Write-Host "✅ 测试完成！" -ForegroundColor Green
Write-Host ""
Write-Host "日志文件位置: test_logs\" -ForegroundColor Yellow
Write-Host "你可以将 test_logs\ 目录中的文件发送给开发者进行修复" -ForegroundColor Yellow
Write-Host ""
Read-Host "按 Enter 键继续"
