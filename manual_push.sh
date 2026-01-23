#!/bin/bash
# 手动推送脚本 - 文档链接修复
# 使用此脚本完成推送到远程仓库

set -e

echo "======================================================================"
echo "  APT-Transformer 文档链接修复 - 手动推送脚本"
echo "======================================================================"
echo ""

# 检查当前目录
if [ ! -d ".git" ]; then
    echo "❌ 错误: 请在 APT-Transformer 项目根目录下运行此脚本"
    exit 1
fi

echo "📍 当前位置: $(pwd)"
echo ""

# 显示待推送的提交
echo "📋 待推送的提交:"
echo "-------------------------------------------------------------------"
git log origin/main..main --oneline --no-decorate
echo "-------------------------------------------------------------------"
echo ""

# 显示统计信息
COMMIT_COUNT=$(git log origin/main..main --oneline | wc -l)
echo "📊 统计: $COMMIT_COUNT 个提交待推送"
echo ""

# 询问用户选择推送方式
echo "请选择推送方式:"
echo ""
echo "  1) 推送到 main 分支 (推荐)"
echo "  2) 推送到feature分支并创建PR"
echo "  3) 应用patch文件 (如果推送失败)"
echo "  4) 查看详细信息后退出"
echo "  5) 取消"
echo ""
read -p "请输入选项 [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 正在推送到 main 分支..."
        git checkout main
        git push origin main

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 成功推送到 main 分支！"
            echo ""
            echo "🔍 验证:"
            git log origin/main -3 --oneline
        else
            echo ""
            echo "❌ 推送失败。请尝试选项 2 或 3"
            exit 1
        fi
        ;;

    2)
        echo ""
        SESSION_ID="wLTkS"
        BRANCH_NAME="claude/fix-documentation-links-${SESSION_ID}"

        echo "📌 创建feature分支: $BRANCH_NAME"
        git checkout -B $BRANCH_NAME

        echo "🚀 正在推送feature分支..."
        git push -u origin $BRANCH_NAME

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 成功推送feature分支！"
            echo ""
            echo "📝 现在可以创建PR:"
            echo ""
            echo "方法1 - 使用 GitHub CLI:"
            echo "  gh pr create \\"
            echo "    --title \"docs: Fix 202 broken documentation links\" \\"
            echo "    --base main \\"
            echo "    --head $BRANCH_NAME \\"
            echo "    --body-file LINK_FIX_SUMMARY.md"
            echo ""
            echo "方法2 - 使用浏览器:"
            echo "  https://github.com/chen0430tw/APT-Transformer/compare/main...$BRANCH_NAME"
        else
            echo ""
            echo "❌ 推送失败。请尝试选项 3"
            exit 1
        fi
        ;;

    3)
        echo ""
        echo "📦 使用patch文件方式..."

        if [ ! -f "link-fixes.patch" ]; then
            echo "⚙️  生成patch文件..."
            git format-patch origin/main --stdout > link-fixes.patch
        fi

        echo ""
        echo "✅ Patch文件已生成: link-fixes.patch"
        echo ""
        echo "📝 应用步骤:"
        echo ""
        echo "1. 在另一台有网络权限的机器上:"
        echo "   git clone https://github.com/chen0430tw/APT-Transformer.git"
        echo "   cd APT-Transformer"
        echo ""
        echo "2. 复制patch文件到该目录"
        echo ""
        echo "3. 应用patch:"
        echo "   git am < link-fixes.patch"
        echo ""
        echo "4. 推送:"
        echo "   git push origin main"
        echo ""

        read -p "是否立即查看patch内容? [y/N]: " view_patch
        if [ "$view_patch" = "y" ] || [ "$view_patch" = "Y" ]; then
            less link-fixes.patch
        fi
        ;;

    4)
        echo ""
        echo "📊 详细信息:"
        echo ""
        echo "=== Git状态 ==="
        git status
        echo ""
        echo "=== 待推送提交详情 ==="
        git log origin/main..main --stat
        echo ""
        echo "=== 远程分支 ==="
        git branch -r
        echo ""
        ;;

    5)
        echo ""
        echo "取消操作"
        exit 0
        ;;

    *)
        echo ""
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "  完成！"
echo "======================================================================"
