# APT Model Debug模式使用指南

## 概述

APT Model现在支持**持久化的Debug模式**，无需每次训练都手动添加`--verbose`参数。Debug模式通过全局配置文件管理，一次启用，永久生效。

---

## 🎯 核心功能

### 1. 全局配置管理
- 配置文件位置：`apt_model/config/settings.yaml`
- 支持的配置项：debug模式、日志级别、训练参数等
- 配置优先级：**环境变量 > 全局配置 > 命令行参数 > 默认值**

### 2. Debug模式切换
- **启用Debug模式**：一次设置，所有后续命令自动启用详细日志
- **禁用Debug模式**：一次设置，恢复正常日志级别
- **临时覆盖**：命令行参数可临时覆盖全局配置

### 3. 系统诊断工具
- 检查Python环境和依赖包
- 验证模型架构和前向传播
- 测试数据加载和分词器
- 生成详细的诊断报告

---

## 🚀 快速开始

### 启用Debug模式

```bash
# 方式1：使用config命令（推荐）
python -m apt_model config --set-debug on

# 方式2：使用环境变量
export APT_DEBUG_ENABLED=1

# 方式3：直接编辑配置文件
# 修改 apt_model/config/settings.yaml
# debug:
#   enabled: true
```

### 禁用Debug模式

```bash
# 禁用Debug模式
python -m apt_model config --set-debug off
```

### 训练时自动使用Debug模式

```bash
# 一旦启用Debug模式，所有训练命令都会自动使用DEBUG日志级别
python -m apt_model train --epochs 10

# 输出示例:
# 🐛 Debug模式已启用 (配置文件: apt_model/config/settings.yaml)
#    日志级别: DEBUG
#    使用 'python -m apt_model config --set-debug off' 可以关闭
```

---

## 📋 配置管理命令

### 查看当前配置

```bash
python -m apt_model config --show
```

输出示例：
```yaml
debug:
  check_gradients: false
  enabled: false
  log_level: INFO
  profile_memory: false
  save_debug_logs: true
hardware:
  auto_gpu: true
  mixed_precision: false
logging:
  colored_output: true
  log_directory: apt_model/log
  log_to_file: true
# ... 更多配置
```

### 设置Debug模式

```bash
# 启用
python -m apt_model config --set-debug on

# 禁用
python -m apt_model config --set-debug off
```

### 获取特定配置

```bash
# 查看debug是否启用
python -m apt_model config --get debug.enabled

# 查看日志级别
python -m apt_model config --get debug.log_level

# 查看默认batch size
python -m apt_model config --get training.default_batch_size
```

### 设置任意配置

```bash
# 设置默认训练epochs
python -m apt_model config --set-key training.default_epochs --set-value 30

# 设置日志级别
python -m apt_model config --set-key debug.log_level --set-value DEBUG

# 启用梯度检查
python -m apt_model config --set-key debug.check_gradients --set-value true
```

### 重置所有配置

```bash
python -m apt_model config --reset
```

---

## 🔧 Debug诊断命令

### 运行全部检查

```bash
python -m apt_model debug
```

输出示例：
```
============================================================
APT Debug Mode - 系统诊断工具
============================================================

[1/4] 检查IO和Python环境...
------------------------------------------------------------
  Python版本: 3.10.12
  工作目录: /home/user/APT-Transformer
  检查PyTorch...
    ✓ PyTorch版本: 2.0.1
    ✓ CUDA可用: True
    ✓ CUDA版本: 11.8
    ✓ GPU数量: 1
  检查必要的包...
    ✓ transformers
    ✓ numpy
    ✓ tqdm

[2/4] 检查模型架构...
------------------------------------------------------------
  加载模型配置...
    ✓ 配置创建成功
  创建模型实例...
    ✓ 模型创建成功
    - 参数数量: 1,234,567
  测试前向传播...
    ✓ 前向传播成功: torch.Size([2, 10, 1000])
  测试生成方法...
    ✓ 生成方法成功: torch.Size([2, 15])

[3/4] 检查数据加载...
------------------------------------------------------------
    ⚠️  未指定数据路径，使用测试数据
    - 第一条: 测试文本1...
    - 平均长度: 5.0
  测试DataLoader...
    ✓ DataLoader正常: 批次大小=2

[4/4] 检查分词器...
------------------------------------------------------------
  测试分词器...
  初始化分词器...
    ✓ 分词器创建成功
    - 检测语言: zh
    - 词汇表大小: 21128
  测试编码解码...
    - 原文: 人工智能
    - 编码: [782, 899, 3255, ...]
    - 解码: 人工智能
    ✓ 编码解码往返一致

============================================================
诊断报告
============================================================
✓ io          : IO流程正常
✓ model       : 模型架构正常
✓ data        : 数据加载正常
✓ tokenizer   : 分词器正常

✓ 所有检查通过！系统运行正常。
```

### 运行特定检查

```bash
# 仅检查IO环境
python -m apt_model debug --type io

# 仅检查模型架构
python -m apt_model debug --type model

# 仅检查数据加载
python -m apt_model debug --type data

# 仅检查分词器
python -m apt_model debug --type tokenizer
```

### 检查自定义数据

```bash
# 检查特定数据文件的加载
python -m apt_model debug --type data --data-path ./my_data.txt
```

---

## 🎨 配置文件说明

配置文件位置：`apt_model/config/settings.yaml`

### Debug配置项

```yaml
debug:
  enabled: false              # 是否启用Debug模式
  log_level: INFO            # 日志级别: DEBUG, INFO, WARNING, ERROR
  profile_memory: false      # 是否进行内存分析
  check_gradients: false     # 是否检查梯度
  save_debug_logs: true      # 是否保存详细的debug日志文件
```

### 训练配置项

```yaml
training:
  default_epochs: 20
  default_batch_size: 8
  default_learning_rate: 3.0e-5
  checkpoint_auto_save: true
```

### 日志配置项

```yaml
logging:
  colored_output: true       # 是否使用彩色日志输出
  log_to_file: true         # 是否同时输出到文件
  log_directory: "apt_model/log"
```

---

## 🌟 使用场景示例

### 场景1：日常开发调试

```bash
# 1. 启用Debug模式
python -m apt_model config --set-debug on

# 2. 运行诊断检查系统状态
python -m apt_model debug

# 3. 开始训练（自动使用DEBUG日志）
python -m apt_model train --epochs 5

# 4. 完成开发后禁用Debug模式
python -m apt_model config --set-debug off
```

### 场景2：问题诊断

```bash
# 当训练出现问题时，运行诊断命令
python -m apt_model debug

# 查看详细的诊断报告，定位问题
# 如果某个检查失败，会显示详细错误信息
```

### 场景3：CI/CD集成

```bash
# 在CI流程中启用debug模式
export APT_DEBUG_ENABLED=1

# 运行测试
python -m apt_model debug
python -m apt_model train --epochs 1 --batch-size 2

# Debug模式会输出详细日志，便于CI调试
```

### 场景4：生产环境

```bash
# 确保Debug模式已禁用
python -m apt_model config --set-debug off

# 查看配置
python -m apt_model config --show | grep debug

# 运行生产训练（使用INFO日志级别）
python -m apt_model train --epochs 100
```

---

## 🔍 优先级说明

Debug模式的优先级从高到低：

1. **命令行参数** `--verbose`
   ```bash
   python -m apt_model train --verbose  # 最高优先级，临时启用
   ```

2. **环境变量** `APT_DEBUG_ENABLED`
   ```bash
   export APT_DEBUG_ENABLED=1
   python -m apt_model train
   ```

3. **全局配置文件** `settings.yaml`
   ```yaml
   debug:
     enabled: true
   ```

4. **默认值** `INFO`

---

## 💡 提示和技巧

### 1. 快速查看Debug状态

```bash
python -m apt_model config --get debug.enabled
```

### 2. 临时启用Debug（不修改配置）

```bash
# 使用命令行参数，仅本次生效
python -m apt_model train --verbose
```

### 3. 使用环境变量（适合CI/CD）

```bash
# 在脚本或CI配置中设置
export APT_DEBUG_ENABLED=1
export APT_DEBUG_LOG_LEVEL=DEBUG
```

### 4. 检查特定模型的问题

```bash
# 指定模型路径进行调试
python -m apt_model debug --type model --model-path ./my_checkpoint
```

### 5. 查看所有可用配置

```bash
python -m apt_model config --show
```

---

## 📚 与旧版本的区别

### 旧版本（每次都要加参数）

```bash
python -m apt_model train --verbose
python -m apt_model chat --verbose
python -m apt_model evaluate --verbose
# 每次都要记得加 --verbose，很麻烦
```

### 新版本（一次设置，永久生效）

```bash
# 一次设置
python -m apt_model config --set-debug on

# 之后所有命令自动启用debug
python -m apt_model train
python -m apt_model chat
python -m apt_model evaluate

# 不需要debug时关闭
python -m apt_model config --set-debug off
```

---

## ❓ 常见问题

### Q1: 如何确认Debug模式已启用？

```bash
python -m apt_model config --get debug.enabled
# 输出: True 表示已启用
```

或者运行任意命令，会显示Debug提示：
```
🐛 Debug模式已启用 (配置文件: apt_model/config/settings.yaml)
   日志级别: DEBUG
   使用 'python -m apt_model config --set-debug off' 可以关闭
```

### Q2: Debug模式会影响训练速度吗？

Debug模式仅影响日志输出量，不会显著影响训练速度。如果担心性能，可以在生产环境禁用。

### Q3: 配置文件在哪里？

配置文件位于：`apt_model/config/settings.yaml`

### Q4: 如何重置所有配置？

```bash
python -m apt_model config --reset
```

### Q5: 可以为不同项目设置不同的配置吗？

目前配置是全局的。如果需要不同配置，可以：
1. 复制整个项目目录
2. 或使用环境变量覆盖：`export APT_DEBUG_ENABLED=1`

---

## 🔗 相关文档

- [原Debug模式检查报告](./debug_mode_analysis.md)
- [配置文件参考](../../apt/core/config/settings.yaml)
- [日志系统文档](../../archived/apt_model/utils/logging_utils.py)

---

## ✨ 总结

新的Debug模式系统提供了：

✅ **持久化配置** - 一次设置，永久生效
✅ **灵活优先级** - 支持环境变量、配置文件、命令行参数
✅ **系统诊断** - 全面的健康检查工具
✅ **易用性** - 简单的命令行界面
✅ **向后兼容** - 保留原有`--verbose`参数

享受更便捷的调试体验！ 🎉
