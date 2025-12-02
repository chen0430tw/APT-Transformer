#!/bin/bash
# APT模型Optuna超参数优化 - 快速测试配置
# 用于快速验证流程，不追求最佳分数

echo "=========================================="
echo "APT模型超参数优化 - 快速测试"
echo "=========================================="
echo
echo "配置说明:"
echo "  - 试验次数: 10次（快速测试）"
echo "  - 每次训练: 3轮（快速验证）"
echo "  - 批次大小: 16（节省显存）"
echo
echo "预计耗时: 约30-60分钟"
echo "=========================================="
echo

# 切换到experiments/hpo目录
cd experiments/hpo || exit 1

# 创建新的时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
STUDY_NAME="apt_optuna_quick_${TIMESTAMP}"
DB_PATH="${STUDY_NAME}.db"

echo "Study名称: ${STUDY_NAME}"
echo "数据库路径: ${DB_PATH}"
echo
echo "开始快速测试..."
echo "=========================================="
echo

# 运行Optuna优化
python apt_optuna_auto.py \
  --trials 10 \
  --epochs 3 \
  --batch-size 16 \
  --study-name "${STUDY_NAME}" \
  --db-path "${DB_PATH}"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo
    echo "=========================================="
    echo "✓ 快速测试完成！"
    echo "=========================================="
    echo
    echo "这只是快速测试，如需获得最佳参数，请运行："
    echo "  ./run_optuna_optimization.sh"
else
    echo
    echo "=========================================="
    echo "✗ 测试出错"
    echo "=========================================="
fi
