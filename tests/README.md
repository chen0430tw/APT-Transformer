# Tests 目录

APT-Transformer 的测试套件。

## 📁 测试文件组织

### 核心测试
- `test_smoke.py` - 快速冒烟测试
- `test_core_imports.py` - 核心导入测试
- `test_console.py` - 控制台功能测试

### 插件系统测试
- `test_plugin_system.py` - 插件系统完整测试
- `test_plugin_system_standalone.py` - 插件系统独立测试
- `test_plugin_version_manager.py` - 插件版本管理测试
- `test_admin_mode_structure.py` - 管理员模式结构测试

### 模型与训练测试
- `test_trainer_complete.py` - 训练器完整测试
- `test_multimodal.py` - 多模态功能测试
- `test_multilingual.py` - 多语言功能测试
- `test_callbacks.py` - 回调函数测试

### 压缩与加速测试
- `test_compression_plugin.py` - 压缩插件测试
- `test_compression_minimal.py` - 最小压缩测试
- `test_compression_plugins.py` - 压缩插件集成测试
- `test_dbc_acceleration.py` - DBC加速测试

### 其他功能测试
- `test_bert_tokenizer.py` - BERT分词器测试
- `test_small_apt_model.py` - 小型APT模型测试
- `test_hlbd_quick_learning.py` - HLBD快速学习测试
- `test_terminator_logic.py` - 终止器逻辑测试
- `test_terminator_scenario.py` - 终止器场景测试
- `test_error_persistence.py` - 错误持久化测试
- `test_legacy_adapters.py` - 旧版适配器测试
- `test_vft_tva.py` - VFT/TVA测试

## 🚀 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行快速冒烟测试
```bash
pytest tests/test_smoke.py -v
```

### 运行特定测试
```bash
pytest tests/test_plugin_system.py -v
```

### 运行特定测试函数
```bash
pytest tests/test_compression_plugin.py::test_dbc_training -v
```

### 运行测试并显示覆盖率
```bash
pytest tests/ --cov=apt_model --cov-report=html
```

## 📊 测试类别

### 🟢 快速测试 (< 1分钟)
- `test_smoke.py`
- `test_core_imports.py`
- `test_compression_minimal.py`

### 🟡 中等测试 (1-5分钟)
- `test_plugin_system.py`
- `test_callbacks.py`
- `test_console.py`

### 🔴 完整测试 (> 5分钟)
- `test_trainer_complete.py`
- `test_multimodal.py`
- `test_compression_plugin.py`

## 🔧 测试配置

配置文件: `conftest.py`
- pytest fixtures
- 测试环境设置
- 共享工具函数

## 💡 编写新测试

### 测试文件命名
```
test_<feature_name>.py
```

### 测试函数命名
```python
def test_<specific_functionality>():
    """测试描述"""
    # 测试代码
```

### 示例
```python
def test_model_forward_pass():
    """测试模型前向传播"""
    model = create_test_model()
    input_data = torch.randn(2, 10)
    output = model(input_data)
    assert output.shape == (2, 5)
```

## 📝 测试最佳实践

1. **每个测试独立**: 不依赖其他测试的结果
2. **清晰的断言**: 使用明确的断言消息
3. **快速测试**: 保持单个测试< 5秒
4. **覆盖边界**: 测试边界情况和异常
5. **文档化**: 添加docstring说明测试目的

## 🐛 调试测试

### 显示print输出
```bash
pytest tests/test_plugin_system.py -s
```

### 进入调试器
```bash
pytest tests/test_plugin_system.py --pdb
```

### 只运行失败的测试
```bash
pytest --lf
```

## 🔗 相关文档

- [项目README](../README.md)
- [插件开发指南](../apt_model/cli/PLUGIN_GUIDE.md)
- [完整文档中心](../docs/README.md)

## 📧 报告问题

如果测试失败，请报告：
1. 失败的测试文件和函数
2. 完整的错误信息
3. 运行环境（Python版本、OS等）
4. 复现步骤

提交Issue: [GitHub Issues](https://github.com/chen0430tw/APT-Transformer/issues)
