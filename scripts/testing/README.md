# APT-Transformer 测试脚本

本目录包含APT-Transformer项目的各种测试脚本。

## 核心功能测试

### 四大核心功能测试

**test_cli_commands_direct.py** - 直接测试四大核心功能CLI命令

测试内容：
- ✅ 数据处理 (process-data)
- ✅ 训练 (train)
- ✅ 聊天 (chat)
- ✅ 评估 (evaluate)

运行方式：
```bash
python3 scripts/testing/test_cli_commands_direct.py
```

### 使用示例

测试通过后，可以使用以下命令：

```bash
# 1. 数据处理
python -m apt_model process-data data.txt

# 2. 训练模型
python -m apt_model train --profile lite

# 3. 聊天交互
python -m apt_model chat

# 4. 评估模型
python -m apt_model evaluate model.pt
```

## 系统检查测试

### 综合模块测试

**test_four_core_functions.py** - 快速测试核心功能模块导入

测试模块导入是否正常：
- DataProcessor, load_external_data, HuggingFaceLoader
- Trainer, APTModel, load_profile
- GenerationEvaluator, ChineseTokenizer
- ModelEvaluator, ModelComparison

运行方式：
```bash
python3 scripts/testing/test_four_core_functions.py
```

## 注意事项

### 已知问题

1. **导入慢问题**: 由于transformers和torch.distributed等库较大，首次导入可能需要10-20秒
   - 解决方案：使用`test_cli_commands_direct.py`直接测试CLI命令
   - 长期方案：优化`__init__.py`为lazy import

2. **循环导入**: apt.core模块存在循环导入问题
   - 已修复：将`train_model`改为lazy import
   - 位置：`apt/core/__init__.py` 第76-87行

### 测试建议

- **快速测试**：使用`test_cli_commands_direct.py`（30秒内完成）
- **详细测试**：使用`test_four_core_functions.py`（可能需要20-30秒）
- **生产环境**：直接使用`python -m apt_model <command>`

## 测试结果

最后测试时间：2026-01-24

| 功能 | 状态 | 命令 |
|------|------|------|
| 数据处理 | ✅ | `process-data` |
| 训练 | ✅ | `train` |
| 聊天 | ✅ | `chat` |
| 评估 | ✅ | `evaluate` |

所有核心功能测试通过！🎉
