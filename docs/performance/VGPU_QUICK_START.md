# 虚拟Blackwell快速启动指南

## 🚀 5分钟开始训练

### 1. 快速测试（Tiny模型）

```bash
cd training
python train_vb_apt.py --config tiny --epochs 5
```

**预期结果**：
- 训练时间：<1分钟
- 显存占用：<500MB
- 适合：验证环境

### 2. 小型训练（Small模型）

```bash
python train_vb_apt.py --config small --epochs 10 --vgpu-auto
```

**配置**：
- 参数量：~7M
- 显存需求：~2GB
- 训练速度：~50ms/batch
- 适合：快速实验

### 3. 中型训练（Medium模型）

```bash
python train_vb_apt.py --config medium --epochs 20 --vgpu-auto --estimate-resources
```

**配置**：
- 参数量：~163M
- 显存需求：~12GB
- 训练速度：~500ms/batch
- 适合：完整训练

### 4. 大型训练（Large模型）

```bash
python train_vb_apt.py --config large --epochs 10 --vgpu-auto --mixed-precision --gradient-checkpointing --estimate-resources
```

**配置**：
- 参数量：~406M
- 显存需求：~8GB（优化后）
- 训练速度：~1000ms/batch
- 适合：生产环境

---

## 📊 命令行参数详解

### 预设配置

| 配置 | 参数量 | 隐藏层 | 层数 | 显存 | 用途 |
|------|--------|--------|------|------|------|
| tiny | 0.3M | 128 | 2 | 0.5GB | 快速测试 |
| small | 7M | 256 | 4 | 2GB | 实验开发 |
| medium | 163M | 768 | 12 | 12GB | 完整训练 |
| large | 406M | 1024 | 24 | 8GB* | 生产部署 |

*使用混合精度+梯度检查点

### 核心参数

```bash
# 基础配置
--config {tiny|small|medium|large}  # 使用预设
--batch-size 16                     # 批次大小
--epochs 10                         # 训练轮数
--learning-rate 1e-4                # 学习率

# VGPU优化
--vgpu-auto                         # 自动配置VGPU堆叠
--use-fp4                           # FP4量化（87.5%显存节省）
--use-flash-attn                    # Flash Attention（35%显存节省）
--mixed-precision                   # 混合精度（50%显存节省）
--gradient-checkpointing            # 梯度检查点（75%激活值节省）

# 资源管理
--estimate-resources                # 训练前评估资源
--save-steps 500                    # 每500步保存
--log-interval 10                   # 每10批次日志
```

### 自定义配置

```bash
python train_vb_apt.py \
    --vocab-size 30000 \
    --hidden-size 512 \
    --num-layers 8 \
    --num-heads 8 \
    --seq-length 1024 \
    --batch-size 16 \
    --epochs 20 \
    --vgpu-auto
```

---

## 💡 使用场景

### 场景1：单卡8GB训练中型模型

```bash
# RTX 3070 / RTX 2080 Ti
python train_vb_apt.py \
    --config medium \
    --vgpu-auto \
    --mixed-precision \
    --gradient-checkpointing \
    --batch-size 4
```

**VGPU配置**：
- Level 0 (GPU): 6GB
- Level 1 (CPU): 6GB
- Level 2 (SSD): 20GB

### 场景2：单卡24GB训练大型模型

```bash
# RTX 3090 / RTX 4090
python train_vb_apt.py \
    --config large \
    --vgpu-auto \
    --mixed-precision \
    --batch-size 8 \
    --use-flash-attn
```

**VGPU配置**：
- Level 0 (GPU): 20GB
- Level 1 (CPU): 8GB
- Level 2 (SSD): 40GB

### 场景3：双卡8GB协同训练

```bash
# 2× RTX 3070
python train_vb_apt.py \
    --config large \
    --vgpu-auto \
    --mixed-precision \
    --gradient-checkpointing \
    --batch-size 4
```

**VGPU配置**：
- Level 0 (GPU 0): 6GB
- Level 1 (GPU 1): 6GB
- Level 2 (CPU): 16GB
- Level 3 (SSD): 40GB

### 场景4：CPU-only训练

```bash
# 无GPU环境
python train_vb_apt.py \
    --config small \
    --batch-size 4 \
    --epochs 5
```

**VGPU配置**：
- Level 0 (CPU): 4GB
- Level 1 (SSD): 16GB

---

## 📈 性能优化技巧

### 1. 批次大小调优

```bash
# 显存充足：增大批次
--batch-size 32

# 显存不足：减小批次
--batch-size 4

# 动态调整：让VGPU自动管理
--vgpu-auto
```

### 2. 序列长度平衡

```bash
# 短序列（快速训练）
--seq-length 512

# 长序列（更好效果）
--seq-length 2048 --use-flash-attn
```

### 3. 梯度累积模拟大批次

```bash
# 4个小批次 = 1个大批次
--batch-size 4 --gradient-accumulation-steps 4
```

### 4. 混合优化组合

```bash
# 最大显存节省
--mixed-precision \
--gradient-checkpointing \
--use-fp4 \
--use-flash-attn

# 预期节省：~90%显存
```

---

## 🔍 监控和调试

### 查看训练日志

```bash
# 实时查看
tail -f output/training_*.log

# 搜索关键信息
grep "Loss" output/training_*.log
grep "VGPU" output/training_*.log
```

### 检查VGPU统计

训练会每5个epoch打印VGPU统计：

```
VGPU堆叠统计:
  Level 0命中率: 95.2%  ← 目标>90%
  提升次数: 145
  降级次数: 23
```

**优化建议**：
- 命中率<80%：增加Level 0容量
- 提升次数多：优化数据访问模式
- 降级次数多：Level 0容量过小

### 性能剖析

```bash
# 启用详细日志
python train_vb_apt.py --config small --log-interval 1

# 分析瓶颈
# - 前向传播慢：考虑use-flash-attn
# - 反向传播慢：考虑gradient-checkpointing
# - 数据加载慢：增加num-workers
```

---

## 📦 输出文件

训练会在`output/`目录生成：

```
output/
├── training_20260120_143022.log    # 训练日志
├── best_model.pt                   # 最佳模型
└── checkpoint_step_500.pt          # 检查点
```

### 加载模型

```python
import torch
from apt_model.modeling.apt_model import APTLargeModel, APTModelConfiguration

# 加载检查点
checkpoint = torch.load('output/best_model.pt')

# 恢复模型
config = APTModelConfiguration(**checkpoint['config'])
model = APTLargeModel(config)
model.load_state_dict(checkpoint['model_state_dict'])

# 推理
model.eval()
output = model(input_ids)
```

---

## 🐛 常见问题

### Q1: OOM (Out of Memory)

**解决**：
```bash
# 方案1：减小批次
--batch-size 2

# 方案2：启用所有优化
--mixed-precision --gradient-checkpointing --vgpu-auto

# 方案3：减小模型
--config small
```

### Q2: 训练速度慢

**检查**：
```bash
# 1. VGPU命中率
grep "Level 0命中率" output/training_*.log

# 2. 批次大小
# 太小会导致GPU利用率低

# 3. 数据加载
# num_workers=0在Windows上最稳定
```

### Q3: 损失不下降

**调试**：
```bash
# 1. 降低学习率
--learning-rate 1e-5

# 2. 检查梯度
--max-grad-norm 1.0

# 3. 增加warmup
# (需要修改代码添加LR scheduler)
```

### Q4: VGPU命中率低

**优化**：
```bash
# 1. 增加Level 0容量
# 编辑vgpu_stack.py中的default_config

# 2. 启用自动配置
--vgpu-auto

# 3. 减小模型规模
--config small
```

---

## 🎯 推荐配置

### 开发环境（快速迭代）

```bash
python train_vb_apt.py \
    --config small \
    --epochs 5 \
    --batch-size 16 \
    --log-interval 5
```

### 实验环境（完整训练）

```bash
python train_vb_apt.py \
    --config medium \
    --epochs 20 \
    --vgpu-auto \
    --mixed-precision \
    --save-steps 1000 \
    --estimate-resources
```

### 生产环境（最优性能）

```bash
python train_vb_apt.py \
    --config large \
    --epochs 50 \
    --vgpu-auto \
    --mixed-precision \
    --gradient-checkpointing \
    --use-flash-attn \
    --save-steps 500 \
    --estimate-resources
```

---

## 📚 相关文档

- [VGPU堆叠架构](VGPU_STACK_ARCHITECTURE.md)
- [GPU Flash优化指南](GPU_FLASH_OPTIMIZATION_GUIDE.txt)
- [成功案例分析](GPU_FLASH_SUCCESS_ANALYSIS.md)

---

*最后更新：2026-01-20*
