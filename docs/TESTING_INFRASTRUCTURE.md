# APT-Transformer 测试基础架构文档

本文档描述项目的自动化测试基础架构，包括所有测试脚本、诊断工具和使用方法。

## 📋 目录

- [快速开始](#快速开始)
- [测试脚本概览](#测试脚本概览)
- [核心测试框架](#核心测试框架)
- [诊断工具](#诊断工具)
- [测试报告](#测试报告)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### Linux/Mac 用户

```bash
# 运行完整测试套件
./scripts/testing/quick_test.sh
```

### Windows 用户

**命令提示符 (CMD):**
```cmd
scripts\testing\quick_test.bat
```

**PowerShell:**
```powershell
.\scripts\testing\quick_test.ps1
```

---

## 📊 测试脚本概览

### 1. 快速测试脚本

| 脚本 | 平台 | 描述 | 测试内容 |
|------|------|------|----------|
| `scripts/testing/quick_test.sh` | Linux/Mac | Bash脚本 | 4个测试套件 |
| `scripts/testing/quick_test.bat` | Windows | 批处理脚本 | 4个测试套件 |
| `scripts/testing/quick_test.ps1` | Windows | PowerShell脚本 | 2个测试套件 |

**测试套件包括:**

1. **基础命令测试** - 测试所有CLI命令
2. **训练后端代码检查** - 检查训练后端语法和依赖
3. **HLBD系统诊断** - 诊断HLBD数据集和配置
4. **生成测试报告** - 生成友好的HTML/文本报告

### 2. 核心测试框架

#### `scripts/testing/test_all_commands.py` (340行)

**功能:**
- 自动测试所有APT Model CLI命令
- 记录测试结果到JSON和日志文件
- 检测导入错误、未知命令、超时等问题

**测试的命令类别:**

```python
# 核心命令 (22个)
CORE_COMMANDS = [
    # 训练相关: train, train-custom, fine-tune, train-hf, train-reasoning, distill
    # 交互相关: chat
    # 评估相关: evaluate, visualize, compare, test
    # 工具相关: clean-cache, estimate, process-data
    # 信息相关: info, list, size
    # 维护相关: prune, backup
    # 分发相关: upload, export-ollama
    # 通用命令: help
]

# Console命令 (10个)
CONSOLE_COMMANDS = [
    "console-status", "console-help", "console-commands",
    "modules-list", "modules-status", "modules-enable",
    "modules-disable", "modules-reload", "debug", "config"
]
```

**输出文件:**
- `test_logs/command_test_<timestamp>.log` - 详细日志
- `test_logs/command_test_<timestamp>.json` - 结构化结果

**使用示例:**

```bash
# 运行所有测试
python scripts/testing/test_all_commands.py

# 查看最新日志
cat test_logs/command_test_*.log | tail -100
```

#### `scripts/testing/view_test_report.py` (190行)

**功能:**
- 解析JSON测试结果
- 生成友好的彩色报告
- 提供根本原因分析和修复建议

**报告内容:**
1. 📊 测试摘要 (总计/通过/失败/跳过/成功率)
2. ❌ 失败的命令 (详细错误信息)
3. ⚠️ 警告的命令
4. ⊘ 跳过的命令
5. ✅ 通过的命令
6. 🔍 根本原因分析 (依赖缺失/未知命令/其他错误)
7. 💡 修复建议

**使用示例:**

```bash
# 查看最新测试报告
python scripts/testing/view_test_report.py

# 查看指定报告
python scripts/testing/view_test_report.py test_logs/command_test_20260122_111322.json
```

---

## 🔧 诊断工具

### `tools/diagnostics/check_training_backends.py` (248行)

**功能:**
训练后端代码检查工具，用于检查新创建的训练后端是否有bug、依赖缺失等问题。

**检查项:**

1. **语法检查** - Python AST解析
2. **依赖检查** - 检测必需和可选依赖
3. **文件引用检查** - 验证文件路径
4. **逻辑检查** - 常见错误模式检测

**检查的文件:**
- `train.py`
- `train_deepspeed.py`
- `train_azure_ml.py`
- `train_hf_trainer.py`

**依赖分类:**

```python
# 标准库（不需要安装）
stdlib = {'os', 'sys', 'json', 'argparse', 'pathlib', ...}

# 必需依赖
required = {'torch', 'numpy'}

# 可选依赖
optional_deps = {
    'deepspeed': 'pip install deepspeed',
    'azure': 'pip install azure-ai-ml mlflow azureml-mlflow',
    'transformers': 'pip install transformers datasets accelerate',
    'wandb': 'pip install wandb',
    'matplotlib': 'pip install matplotlib',
    'datasets': 'pip install datasets',
    'mlflow': 'pip install mlflow',
}
```

**使用示例:**

```bash
python tools/diagnostics/check_training_backends.py
```

### `tools/diagnostics/diagnose_issues.py` (305行)

**功能:**
APT项目问题诊断和修复报告，自动检查所有潜在问题。

**诊断检查:**

1. **依赖检查** - Python包依赖
   - 必需依赖: torch, json, pathlib
   - 可视化依赖: numpy, matplotlib
   - 可选依赖: datasets

2. **Weight Decay检查** - HLBD脚本配置
   - 检查优化器配置
   - 检测是否设置weight_decay参数

3. **HLBD数据集检查** - 数据集完整性
   - 验证 `HLBD_Hardcore_Full.json`
   - 检查反向学英文数据
   - 统计各模块数据量

4. **HLBD验证功能检查** - 测试函数
   - test_generation() 函数
   - evaluate_hlbd_model() 函数
   - 独立验证脚本

5. **潜在Bug检查**
   - 可视化脚本的numpy依赖
   - 训练脚本的checkpoint恢复功能

**自动生成修复脚本:**
运行后会生成 `fix_issues.sh`，包含修复所有问题的命令。

**使用示例:**

```bash
# 运行诊断
python tools/diagnostics/diagnose_issues.py

# 执行自动生成的修复脚本
bash fix_issues.sh
```

---

## 📈 测试报告

### 报告示例

```
================================================================================
APT Model 命令测试报告
================================================================================

📊 测试摘要
   时间: 2026-01-22T11:13:22
   总计: 32 个命令
   ✓ 通过: 25
   ✗ 失败: 5
   ⊘ 跳过: 2
   成功率: 83.3%

--------------------------------------------------------------------------------
📋 详细结果

❌ 失败的命令:
   • evaluate
     错误: Missing dependencies
     退出码: 1
     详情: ModuleNotFoundError: No module named 'torch'

   • visualize
     错误: Missing dependencies
     退出码: 1
     详情: ModuleNotFoundError: No module named 'numpy'

⚠️  警告的命令:
   • estimate: Non-zero exit code (might be expected)

⊘ 跳过的命令:
   • train: Interactive or long-running
   • chat: Interactive or long-running

✅ 通过的命令:
   • help (0.32s)
   • info (0.45s)
   • list (0.38s)
   • console-status (0.41s)
   • ...

--------------------------------------------------------------------------------
🔍 根本原因分析

   缺失依赖:
      • torch
      • numpy
      • matplotlib

   未知命令:
      • (无)

   其他错误:
      • estimate

--------------------------------------------------------------------------------
💡 修复建议

   1. 安装 PyTorch 和相关依赖:
      pip install torch transformers

   2. 安装完整依赖:
      pip install -r requirements.txt

--------------------------------------------------------------------------------

完整日志: test_logs/command_test_20260122_111322.log
文本日志: test_logs/command_test_20260122_111322.json
```

---

## 🔍 常见问题

### Q1: 所有命令都失败，提示ModuleNotFoundError

**原因:** 缺少Python依赖（torch, numpy等）

**解决方案:**

```bash
# 安装核心依赖
pip install torch numpy

# 或安装完整依赖
pip install -r requirements.txt
```

### Q2: 某些命令被跳过

**原因:** 这些命令是交互式的或需要长时间运行（如train, chat）

**说明:** 这是正常行为，测试框架会自动跳过这些命令以避免测试卡住。

### Q3: 测试报告保存在哪里？

**位置:** `test_logs/` 目录

**文件格式:**
- `command_test_<timestamp>.log` - 详细文本日志
- `command_test_<timestamp>.json` - 结构化JSON结果

### Q4: 如何只测试特定命令？

**方法1: 修改test_all_commands.py**

编辑 `CORE_COMMANDS` 或 `CONSOLE_COMMANDS` 列表，只保留需要测试的命令。

**方法2: 直接运行单个命令**

```bash
python -m apt_model <command> --help
```

### Q5: 测试超时怎么办？

**默认超时:** 30秒

**修改超时时间:**

编辑 `scripts/testing/test_all_commands.py` 第169行：

```python
process = subprocess.run(
    cmd_parts,
    capture_output=True,
    text=True,
    timeout=30,  # 修改这里，单位：秒
    encoding='utf-8',
    errors='replace'
)
```

### Q6: 如何添加新的测试命令？

**步骤:**

1. 打开 `scripts/testing/test_all_commands.py`
2. 将新命令添加到 `CORE_COMMANDS` 或 `CONSOLE_COMMANDS` 列表
3. 如果命令需要特殊参数，添加到 `COMMAND_ARGS`
4. 如果命令只需要测试 `--help`，添加到 `HELP_ONLY_COMMANDS`
5. 如果命令需要跳过，添加到 `SKIP_COMMANDS`

**示例:**

```python
# 添加新命令
CORE_COMMANDS = [
    # ... 现有命令 ...
    "my-new-command",
]

# 如果需要特殊参数
COMMAND_ARGS = {
    "my-new-command": ["--arg1", "value1"],
}

# 如果只测试help
HELP_ONLY_COMMANDS = {
    "my-new-command",
}
```

---

## 📚 相关文档

- [CLI命令增强文档](CLI_ENHANCEMENTS.md) - Profile、Pipeline、模块选择
- [高级CLI命令文档](ADVANCED_CLI_COMMANDS.md) - MoE、Blackwell、AIM、NPU、RAG、MXFP4
- [代码检查报告](../archived/reports/CODE_CHECK_REPORT.md) - 综合代码质量检查结果
- [训练后端指南](performance/TRAINING_BACKENDS.md) - 训练后端使用指南
- [可视化指南](product/VISUALIZATION_GUIDE.md) - 可视化使用指南

---

## 🔄 测试工作流

### 开发工作流

```
1. 开发新功能/修复bug
   ↓
2. 运行quick_test.sh
   ↓
3. 检查测试报告
   ↓
4. 如有失败 → 修复 → 回到步骤2
   ↓
5. 所有测试通过 → 提交代码
```

### CI/CD集成

```yaml
# .github/workflows/test.yml 示例
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          ./scripts/testing/quick_test.sh

      - name: Upload test reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: test_logs/
```

---

## 📝 维护指南

### 定期维护任务

1. **每周:**
   - 运行完整测试套件
   - 检查新增命令是否已添加到测试列表
   - 更新 `SKIP_COMMANDS` 列表

2. **每月:**
   - 清理旧测试日志（`test_logs/` 目录）
   - 检查依赖更新
   - 更新测试文档

3. **每次重大重构:**
   - 运行诊断工具
   - 更新测试脚本中的命令列表
   - 验证所有路径引用

### 清理测试日志

```bash
# 删除30天前的日志
find test_logs/ -name "*.log" -mtime +30 -delete
find test_logs/ -name "*.json" -mtime +30 -delete

# 只保留最新10个日志
ls -t test_logs/*.log | tail -n +11 | xargs rm -f
ls -t test_logs/*.json | tail -n +11 | xargs rm -f
```

---

## 🛠️ 故障排查

### 问题: quick_test.sh 权限被拒绝

**解决方案:**

```bash
chmod +x scripts/testing/quick_test.sh
```

### 问题: Windows上路径不正确

**原因:** Windows使用反斜杠 `\`，而脚本使用正斜杠 `/`

**解决方案:** 使用 `.bat` 脚本（已修复路径）

### 问题: Python模块导入错误

**检查步骤:**

1. 验证Python版本: `python --version` (需要3.7+)
2. 检查PYTHONPATH: `echo $PYTHONPATH`
3. 验证项目根目录在PYTHONPATH中
4. 重新安装依赖: `pip install -r requirements.txt`

---

## 📊 测试覆盖率

### 当前覆盖情况

| 模块 | 单元测试 | 集成测试 | CLI测试 |
|------|----------|----------|---------|
| L0 Core | ✅ | ✅ | ✅ |
| L1 Performance | ✅ | ✅ | ✅ |
| L2 Memory | ⚠️ | ✅ | ✅ |
| L3 Product | ✅ | ✅ | ✅ |
| Plugins | ⚠️ | ⚠️ | ✅ |
| CLI | N/A | N/A | ✅ |

图例:
- ✅ 良好覆盖 (>80%)
- ⚠️ 部分覆盖 (50-80%)
- ❌ 缺少覆盖 (<50%)

---

## 🎯 未来改进

### 短期目标 (1-3个月)

- [ ] 添加单元测试覆盖率报告
- [ ] 集成pytest框架
- [ ] 添加性能基准测试
- [ ] 实现自动回归测试

### 中期目标 (3-6个月)

- [ ] 添加E2E测试
- [ ] 实现测试并行化
- [ ] 集成代码覆盖率工具
- [ ] 添加性能监控

### 长期目标 (6-12个月)

- [ ] 完全自动化测试流程
- [ ] 建立测试数据管理系统
- [ ] 实现智能测试选择
- [ ] 添加视觉回归测试

---

## 📞 获取帮助

如有问题或建议，请:

1. 查看本文档的[常见问题](#常见问题)部分
2. 查看项目的 [CONTRIBUTING.md](../CONTRIBUTING.md)
3. 提交Issue到GitHub仓库
4. 联系维护团队

---

**文档版本:** 1.0.0
**最后更新:** 2026-01-22
**维护者:** APT-Transformer Team
