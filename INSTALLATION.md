# APT-Transformer 安装指南

## 问题说明

如果你遇到 `No module named apt_model` 错误，这是因为 Python 无法找到 `apt_model` 模块。这通常发生在：

1. 直接运行 `python -m apt_model` 而没有安装包
2. 在错误的目录下运行命令
3. PYTHONPATH 没有正确设置

## 解决方案

### 方法 1：开发模式安装（推荐）

这是最简单且最推荐的方法，适合开发和使用：

```bash
# 进入项目根目录
cd APT-Transformer

# 安装为可编辑包（开发模式）
pip install -e .

# 验证安装
python -m apt_model --help
# 或者
apt-model --help
```

**优点：**
- 可以在任何目录下运行 `python -m apt_model`
- 可以使用 `apt-model` 命令直接运行
- 代码修改会立即生效，无需重新安装
- 符合 Python 包管理的最佳实践

### 方法 2：在项目根目录下运行

如果不想安装包，可以确保在项目根目录下运行：

```bash
# 确保你在 APT-Transformer 目录下
cd /path/to/APT-Transformer

# 运行命令
python -m apt_model train --epochs 10 --batch-size 8
```

### 方法 3：设置 PYTHONPATH（不推荐）

```bash
# Windows
set PYTHONPATH=C:\path\to\APT-Transformer;%PYTHONPATH%

# Linux/Mac
export PYTHONPATH=/path/to/APT-Transformer:$PYTHONPATH

# 运行命令
python -m apt_model train
```

## 完整安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/chen0430tw/APT-Transformer.git
cd APT-Transformer
```

### 2. 创建虚拟环境（可选但推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装 apt_model 包（开发模式）
pip install -e .
```

### 4. 验证安装

```bash
# 方法 1：使用 python -m
python -m apt_model --help

# 方法 2：使用命令行工具
apt-model --help

# 方法 3：Python 导入测试
python -c "from apt_model.main import main; print('Import successful!')"
```

## 常见问题

### Q1: `pip install -e .` 失败

**错误信息：** `ERROR: File "setup.py" or "setup.cfg" not found`

**解决方案：** 确保你在 APT-Transformer 根目录下，该目录应该包含 `setup.py` 文件。

### Q2: 仍然显示 `No module named apt_model`

**检查步骤：**

```bash
# 1. 检查包是否安装
pip list | grep apt-model

# 2. 检查 Python 可以找到的路径
python -c "import sys; print('\n'.join(sys.path))"

# 3. 检查当前目录
pwd  # Linux/Mac
cd   # Windows

# 4. 重新安装
pip uninstall apt-model
pip install -e .
```

### Q3: Windows 下的路径问题

如果在 Windows 下遇到路径问题：

```bash
# 使用正确的 Python 可执行文件
python -m apt_model train

# 如果有多个 Python 版本，指定完整路径
C:\Python312\python.exe -m apt_model train

# 或者使用 py 启动器
py -3.12 -m apt_model train
```

### Q4: 模块导入错误

如果遇到其他模块导入错误（如 `No module named apt_model.cli.parser`）：

```bash
# 确认所有依赖已安装
pip install -r requirements.txt

# 检查 __init__.py 文件是否存在
find apt_model -name "__init__.py"  # Linux/Mac
dir /s /b apt_model\__init__.py     # Windows
```

## 使用示例

安装完成后，你可以使用以下命令：

### 训练模型

```bash
# 基础训练
python -m apt_model train --epochs 10 --batch-size 8

# 使用自定义数据
python -m apt_model train --data-path ./data/train.txt --epochs 20

# 强制使用 CPU
python -m apt_model train --force-cpu
```

### 启动 WebUI

```bash
python -m apt_model.webui.app
```

### 交互式聊天

```bash
python -m apt_model chat
```

### 查看帮助

```bash
# 查看所有命令
python -m apt_model console-commands

# 查看特定命令帮助
python -m apt_model console-help train
```

## 开发者注意事项

如果你要修改 `apt_model` 的代码：

1. 使用 `pip install -e .` 安装为开发模式
2. 修改代码后无需重新安装
3. 如果修改了 `setup.py`，需要重新运行 `pip install -e .`
4. 如果添加了新的依赖，更新 `requirements.txt` 后运行 `pip install -r requirements.txt`

## 卸载

如果需要卸载 apt-model：

```bash
pip uninstall apt-model
```

## 更多帮助

如果问题仍未解决，请：

1. 检查 [README.md](README.md) 中的详细文档
2. 查看 [GitHub Issues](https://github.com/chen0430tw/APT-Transformer/issues)
3. 提交新的 Issue 并附上错误信息和环境信息：
   ```bash
   python --version
   pip list
   ```
