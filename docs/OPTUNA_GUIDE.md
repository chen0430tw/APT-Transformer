# APT模型Optuna超参数优化指南

## 📋 概述

本指南提供了使用Optuna进行APT模型超参数优化的配置和使用方法。

## 🚀 快速开始

### 方案1: 推荐配置（深度优化）

**目标**: 获得最佳超参数，超越上次的74.67/100分数

```bash
./run_optuna_optimization.sh
```

**配置详情**:
- 试验次数: 100次
- 每次训练轮数: 10轮
- 批次大小: 32
- 预计耗时: 5-10小时（取决于硬件）

### 方案2: 快速测试

**目标**: 快速验证流程是否正常

```bash
./run_optuna_quick_test.sh
```

**配置详情**:
- 试验次数: 10次
- 每次训练轮数: 3轮
- 批次大小: 16
- 预计耗时: 30-60分钟

---

## 🎯 优化的超参数范围

| 参数 | 范围 | 说明 | 重要性 |
|------|------|------|--------|
| base_lr | 2e-5 ~ 8e-5 | 基础学习率 | ⭐⭐⭐ 最重要(90%) |
| alpha | 0.0005 ~ 0.002 | 学习率调整系数 | ⭐⭐ 次重要(3.4%) |
| init_tau | 1.0 ~ 1.8 | 初始温度 | ⭐ 第三重要(2.8%) |
| epsilon | 0.05 ~ 0.15 | Taylor展开系数 | 影响较小(<1%) |
| beta | 0.001 ~ 0.008 | 动量系数 | 影响较小(<1%) |
| dropout | 0.1 ~ 0.25 | Dropout率 | 影响较小(<1%) |
| attention_dropout | 0.1 ~ 0.25 | 注意力Dropout | 影响较小(<1%) |
| sr_ratio | 4 ~ 8 | 空间缩减比例 | 影响很小(<0.2%) |
| weight_decay | 0.01 ~ 0.03 | 权重衰减 | 影响较小(<1%) |
| gradient_clip | 0.5 ~ 1.2 | 梯度裁剪 | 影响较小(<1%) |

**重要发现**: 根据上次优化，**学习率(base_lr)** 对模型性能的影响占90%以上！

---

## 📊 上次优化结果回顾

- **优化时间**: 2025-03-10 16:27:05
- **试验次数**: 50次
- **最佳分数**: **74.67/100**
- **最佳Trial**: #23

**最佳超参数**:
```python
base_lr = 2.418e-05
epsilon = 0.084046
alpha = 0.001065
beta = 0.002227
dropout = 0.150236
attention_dropout = 0.194551
sr_ratio = 6
init_tau = 1.348287
weight_decay = 0.012845
gradient_clip = 0.945565
```

---

## 🛠️ 自定义配置

如果需要自定义参数，直接运行：

```bash
cd experiments/hpo

# 自定义配置示例
python apt_optuna_auto.py \
  --trials 50 \           # 试验次数
  --epochs 8 \            # 每次训练轮数
  --batch-size 24 \       # 批次大小
  --study-name my_study \ # Study名称
  --db-path my_study.db   # 数据库路径
```

### 可选参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trials` | 50 | 优化试验次数 |
| `--epochs` | 5 | 每次试验的训练轮数 |
| `--batch-size` | 32 | 训练批次大小 |
| `--study-name` | 自动生成 | Optuna Study名称 |
| `--db-path` | 自动生成 | SQLite数据库路径 |
| `--python-path` | python | Python解释器路径 |

---

## 📁 输出文件说明

优化完成后会在 `experiments/hpo/` 目录生成以下文件：

```
experiments/hpo/
├── best_apt_config_TIMESTAMP.py      # 最佳配置文件
├── best_train_cmd_TIMESTAMP.sh       # 训练脚本（Linux/macOS）
├── best_train_cmd_TIMESTAMP.bat      # 训练脚本（Windows）
├── optuna_results_TIMESTAMP.txt      # 详细优化报告
├── optuna_history_TIMESTAMP.png      # 优化历史图表
├── optuna_importance_TIMESTAMP.png   # 参数重要性图表
└── apt_optuna_TIMESTAMP.db           # Optuna数据库
```

---

## 🎓 使用最佳参数

### 方法1: 使用生成的脚本

```bash
cd experiments/hpo
bash best_train_cmd_TIMESTAMP.sh
```

### 方法2: 手动应用配置

```bash
# 1. 复制最佳配置
cp experiments/hpo/best_apt_config_TIMESTAMP.py apt_model/config/apt_config.py

# 2. 运行训练
python -m apt_model train \
  --epochs 20 \
  --batch-size 8 \
  --learning-rate 2.418e-05 \
  --save-path apt_model_best
```

---

## 💡 优化技巧

### 1. 继续之前的优化

```bash
python apt_optuna_auto.py \
  --study-name apt_optuna_20250310_1602 \
  --db-path apt_optuna_20250310_1602.db \
  --trials 50  # 再跑50次
```

### 2. 显存不足的解决方案

脚本会自动：
- 检测可用显存
- 使用梯度累积（自动调整步数）
- 必要时减小批次大小
- 显存仍不足则跳过该试验

### 3. 并行优化（如果有多GPU）

```bash
# Terminal 1
python apt_optuna_auto.py --study-name shared_study --db-path shared.db --trials 25

# Terminal 2 (同时运行)
python apt_optuna_auto.py --study-name shared_study --db-path shared.db --trials 25
```

---

## 🐛 常见问题

### Q: 优化过程中断了怎么办？
A: 使用相同的 `--study-name` 和 `--db-path` 重新运行即可继续。

### Q: 如何查看优化进度？
A: 实时查看日志：
```bash
tail -f experiments/hpo/optuna_temp/train_log_trial_*.txt
```

### Q: 分数一直没有提升怎么办？
A:
1. 检查是否陷入局部最优（试验次数太少）
2. 考虑调整参数搜索范围
3. 增加每次试验的训练轮数（epochs）

### Q: 如何查看历史所有优化结果？
A:
```bash
ls -lht experiments/hpo/optuna_results_*.txt
cat experiments/hpo/optuna_results_TIMESTAMP.txt
```

---

## 📞 技术支持

- 查看详细日志: `experiments/hpo/optuna_temp/`
- Optuna文档: https://optuna.readthedocs.io/
- APT模型配置: `apt_model/config/apt_config.py`

---

**Good luck with your optimization! 🚀**
