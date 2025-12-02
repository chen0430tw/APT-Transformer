#!/bin/bash
echo "=========================================="
echo "使用Optuna优化后的最佳参数进行训练"
echo "优化分数: 74.67/100"
echo "=========================================="
echo

echo "步骤1: 备份当前配置..."
cp apt_model/config/apt_config.py apt_model/config/apt_config.py.backup 2>/dev/null || echo "  (无需备份)"

echo "步骤2: 应用最佳配置..."
cp experiments/configs/best/best_apt_config_20250310_162705.py apt_model/config/apt_config.py
echo "  ✓ 配置文件已更新"
echo

echo "步骤3: 开始训练..."
echo "  训练参数："
echo "    - Epochs: 20"
echo "    - Batch Size: 8"
echo "    - Learning Rate: 2.418e-05"
echo "    - 保存路径: apt_model_best"
echo

export APT_NO_STDOUT_ENCODING=1

python -m apt_model train \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 2.418394206230888e-05 \
  --save-path apt_model_best

echo
echo "=========================================="
echo "训练完成！"
echo "模型已保存到: apt_model_best/"
echo "=========================================="
