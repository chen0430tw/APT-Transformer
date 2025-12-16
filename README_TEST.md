# APT Model 自动化命令测试

## 简介

这个自动化测试工具可以帮你快速测试所有 APT Model 的命令，并生成详细的错误日志。

## 使用方法

### 1. 运行测试

```bash
# 运行所有命令测试
python test_all_commands.py
```

### 2. 查看结果

测试完成后会在 `test_logs/` 目录下生成两个文件：

- `command_test_YYYYMMDD_HHMMSS.log` - 详细的文本日志
- `command_test_YYYYMMDD_HHMMSS.json` - 结构化的 JSON 结果

### 3. 发送日志

将生成的日志文件（`.log` 或 `.json`）发送给开发者即可一次性修复所有问题。

## 测试范围

### 核心命令
- **训练相关**: train, train-custom, fine-tune, train-hf, train-reasoning, distill
- **交互相关**: chat
- **评估相关**: evaluate, visualize, compare, test
- **工具相关**: clean-cache, estimate, process-data
- **信息相关**: info, list, size
- **维护相关**: prune, backup
- **分发相关**: upload, export-ollama
- **通用命令**: help

### Console 命令
- console-status
- console-help
- console-commands
- modules-list
- modules-status
- modules-enable
- modules-disable
- modules-reload
- debug
- config

## 测试策略

1. **跳过长时间命令**: 训练相关命令会被跳过（标记为 SKIPPED）
2. **帮助测试**: 部分命令只测试 `--help` 参数
3. **快速检查**: 每个命令最多运行30秒后自动超时
4. **智能判断**: 区分真正的错误和预期的参数缺失错误

## 日志示例

### 文本日志 (.log)
```
[2025-12-16 10:30:00] [INFO] Testing command: python -m apt_model help
[2025-12-16 10:30:01] [SUCCESS] ✓ PASSED: help
[2025-12-16 10:30:01] [INFO] ⊘ SKIPPED: train - Interactive or long-running
[2025-12-16 10:30:02] [ERROR] ✗ FAILED: evaluate - Missing dependencies
```

### JSON 结果
```json
{
  "summary": {
    "total": 30,
    "passed": 20,
    "failed": 5,
    "skipped": 5
  },
  "results": [
    {
      "command": "help",
      "status": "passed",
      "exit_code": 0,
      "duration": 1.2
    }
  ]
}
```

## 自定义测试

如果需要自定义测试行为，可以修改 `test_all_commands.py` 中的配置：

```python
# 跳过特定命令
SKIP_COMMANDS = {
    "chat",
    "train",
}

# 为命令添加参数
COMMAND_ARGS = {
    "console-help": ["train"],
}

# 只测试 --help
HELP_ONLY_COMMANDS = {
    "evaluate",
}
```

## 故障排查

如果测试脚本本身无法运行：

1. 检查 Python 环境：`python --version` (需要 3.8+)
2. 检查项目路径是否正确
3. 查看是否有基本的依赖问题

## 输出文件位置

```
APT-Transformer/
├── test_all_commands.py       # 测试脚本
├── README_TEST.md             # 本说明文件
└── test_logs/                 # 测试结果目录
    ├── command_test_20251216_103000.log
    └── command_test_20251216_103000.json
```
