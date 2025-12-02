#!/bin/bash
# APT模型Optuna超参数优化 - 推荐配置
# 目标：超越上次的74.67/100分数

echo "=========================================="
echo "APT模型超参数优化 - 推荐配置"
echo "=========================================="
echo
echo "优化目标: 超越上次最佳分数 74.67/100"
echo
echo "配置说明:"
echo "  - 试验次数: 100次（更充分的搜索空间）"
echo "  - 每次训练: 10轮（充分评估参数效果）"
echo "  - 批次大小: 32（平衡速度和效果）"
echo "  - 使用全新数据库（重新开始）"
echo
echo "预计耗时: 根据硬件，约5-10小时"
echo "=========================================="
echo

# 切换到experiments/hpo目录
cd experiments/hpo || exit 1

# 创建新的时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
STUDY_NAME="apt_optuna_${TIMESTAMP}"
DB_PATH="${STUDY_NAME}.db"

echo "Study名称: ${STUDY_NAME}"
echo "数据库路径: ${DB_PATH}"
echo
echo "开始优化..."
echo "=========================================="
echo

# 运行Optuna优化
python apt_optuna_auto.py \
  --trials 100 \
  --epochs 10 \
  --batch-size 32 \
  --study-name "${STUDY_NAME}" \
  --db-path "${DB_PATH}"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo
    echo "=========================================="
    echo "✓ 优化完成！"
    echo "=========================================="
    echo
    echo "生成的文件："
    ls -lh best_apt_config_*.py best_train_cmd_*.* optuna_*.png 2>/dev/null | tail -5
    echo
    echo "下一步："
    echo "  1. 查看优化结果: cat optuna_results_${TIMESTAMP}.txt"
    echo "  2. 查看图表: optuna_history_*.png, optuna_importance_*.png"
    echo "  3. 使用最佳参数训练: bash best_train_cmd_*.sh"
else
    echo
    echo "=========================================="
    echo "✗ 优化过程出错"
    echo "=========================================="
    echo "请检查日志输出"
fi
