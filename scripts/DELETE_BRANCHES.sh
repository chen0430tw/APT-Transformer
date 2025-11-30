#!/bin/bash

echo "🗑️  删除已合并的远程分支..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 要删除的分支列表
branches=(
    "codex"
    "ta7zpi-codex/summarize-document-content"
    "claude/hello-world-011CUQ2B9rjmQ1iNFb5jqNNK"
    "claude/merge-all-branches-011CUQ2B9rjmQ1iNFb5jqNNK"
    "d28cxz-codex/summarize-code-branch-file-structure"
)

# 删除每个分支
for branch in "${branches[@]}"; do
    echo "正在删除: $branch"
    if git push origin --delete "$branch" 2>&1; then
        echo "✅ 成功删除: $branch"
    else
        echo "❌ 删除失败: $branch"
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "清理本地分支引用..."
git branch -D claude/hello-world-011CUQ2B9rjmQ1iNFb5jqNNK 2>/dev/null && echo "✅ 删除本地分支" || echo "ℹ️  本地分支已不存在"
git branch -D claude/merge-all-branches-011CUQ2B9rjmQ1iNFb5jqNNK 2>/dev/null && echo "✅ 删除本地分支" || echo "ℹ️  本地分支已不存在"
git fetch --prune
echo ""

echo "🎉 完成！验证剩余分支..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git branch -a
