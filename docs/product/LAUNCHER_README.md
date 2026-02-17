# APT Transformer 启动器使用指南

## 📋 概述

APT Launcher是一个图形界面启动器，让您无需打开终端即可使用APT Transformer的所有功能。

**主要功能:**
- ✅ 训练模型
- ✅ 微调模型
- ✅ 对话测试
- ✅ 模型评估
- ✅ Optuna超参数优化
- ✅ Debug模式管理

---

## 🚀 快速开始

### 方法1: 双击启动（推荐）

#### Windows用户
1. 双击 `APT_Launcher.bat`
2. 等待GUI界面启动

#### Linux/macOS用户
1. 双击 `APT_Launcher.sh`
2. 如果无法双击启动，在终端运行:
   ```bash
   bash APT_Launcher.sh
   ```

### 方法2: Python直接启动

```bash
# Windows
pythonw apt_launcher.pyw

# Linux/macOS
python3 apt_launcher.pyw
```

---

## 🖥️ 创建桌面快捷方式

运行以下命令在桌面创建快捷方式：

```bash
python create_desktop_shortcut.py
```

**Windows用户注意:**
- 如果提示缺少pywin32，请运行: `pip install pywin32`

**Linux用户注意:**
- 首次使用可能需要右键快捷方式 -> 属性 -> 勾选"允许作为程序执行"

**macOS用户注意:**
- 首次运行可能需要在"系统偏好设置 -> 安全性与隐私"中允许

---

## 📚 功能说明

### 1. 训练模型

在"训练模型"标签页中：

**参数设置:**
- **训练轮数 (Epochs)**: 默认10，根据数据量调整
- **批次大小 (Batch Size)**: 默认8，显存不足可减小
- **学习率 (Learning Rate)**: 默认0.0001
- **保存路径**: 训练后的模型保存位置

**操作步骤:**
1. 设置训练参数
2. 点击"开始训练"
3. 在日志窗口查看训练进度
4. 训练完成后模型保存到指定路径

### 2. 微调模型

在"微调模型"标签页中：

**参数设置:**
- **预训练模型路径**: 要微调的模型位置
- **训练数据路径**: 微调数据文件（.txt格式，每行一个样本）
- **训练轮数**: 微调建议3-10轮
- **学习率**: 微调建议1e-5到5e-5
- **冻结Embedding层**: 勾选可减少训练参数，防止过拟合

**使用场景:**
- 领域适应（将通用模型适配到特定领域）
- 任务专精（针对特定任务优化）
- 参数高效微调（冻结大部分层，只训练顶层）

**操作步骤:**
1. 点击"浏览"选择预训练模型
2. 点击"浏览"选择训练数据
3. 设置微调参数
4. 可选：勾选"冻结Embedding层"
5. 点击"开始微调"

### 3. 对话测试

在"对话测试"标签页中：

**功能:**
- 与训练好的模型进行交互式对话
- 支持多轮对话
- 模型会记住上下文

**操作步骤:**
1. 选择模型路径
2. 点击"启动聊天界面"
3. 在新打开的终端窗口中输入文本
4. 输入 `quit` 或 `exit` 退出对话

### 4. 模型评估

在"模型评估"标签页中：

**功能:**
- 评估模型生成质量
- 生成样本文本
- 计算质量评分

**操作步骤:**
1. 选择要评估的模型路径
2. 点击"开始评估"
3. 在日志窗口查看评估结果和生成样本

### 5. 工具

在"工具"标签页中：

#### Optuna超参数优化

**快速测试 (10试验, 3轮):**
- 用途: 快速验证流程
- 耗时: 约30-60分钟
- 点击"运行快速测试"

**深度优化 (100试验, 10轮):**
- 用途: 获得最佳超参数
- 耗时: 约5-10小时
- 点击"运行深度优化"

#### Debug模式

- **开启Debug模式**: 启用详细日志输出
- **关闭Debug模式**: 禁用详细日志

---

## ⚙️ 系统要求

### 最低要求

- **操作系统**: Windows 7+, macOS 10.12+, Linux (任何主流发行版)
- **Python**: 3.8 或更高版本
- **内存**: 4GB RAM
- **磁盘**: 2GB 可用空间

### 推荐配置

- **操作系统**: Windows 10+, macOS 11+, Ubuntu 20.04+
- **Python**: 3.10 或更高版本
- **内存**: 16GB RAM
- **GPU**: NVIDIA GPU with 8GB+ VRAM（可选，用于加速训练）
- **磁盘**: 10GB+ 可用空间（用于存储模型和数据）

### 依赖检查

GUI启动器需要以下Python包：
- `tkinter` (通常随Python自带)

**如果缺少tkinter:**

Windows:
```bash
# 重新安装Python时确保勾选tk/tcl组件
```

Ubuntu/Debian:
```bash
sudo apt-get install python3-tk
```

Fedora/CentOS:
```bash
sudo dnf install python3-tkinter
```

macOS:
```bash
# 通常自带，如缺失请重新安装Python
brew install python-tk
```

---

## 🐛 常见问题

### Q: 双击启动器没有反应？

**Windows:**
1. 右键 `APT_Launcher.bat` -> 以管理员身份运行
2. 检查Python是否在系统PATH中
3. 尝试手动运行: `pythonw apt_launcher.pyw`

**Linux:**
1. 给脚本添加执行权限: `chmod +x APT_Launcher.sh`
2. 右键 -> 属性 -> 权限 -> 勾选"允许作为程序执行"
3. 或在终端运行: `bash APT_Launcher.sh`

**macOS:**
1. 在终端运行: `bash APT_Launcher.sh`
2. 如提示安全限制，在"系统偏好设置 -> 安全性与隐私"中允许

### Q: 提示缺少tkinter？

**解决方法:**
- Windows: 重新安装Python，确保勾选"tcl/tk and IDLE"
- Linux: `sudo apt-get install python3-tk` (Debian/Ubuntu) 或 `sudo dnf install python3-tkinter` (Fedora)
- macOS: `brew install python-tk`

### Q: 训练/微调时显存不足？

**解决方法:**
1. 减小批次大小（Batch Size）: 8 -> 4 -> 2
2. 微调时勾选"冻结Embedding层"
3. 在CLI中使用 `--gradient-accumulation-steps` 参数

### Q: 聊天界面在哪里？

聊天功能会在**新的终端窗口**中打开，不在GUI界面内。

### Q: 如何查看详细日志？

1. 启用Debug模式（在"工具"标签页）
2. 或查看项目目录下的日志文件

---

## 💡 使用技巧

### 1. 快速训练小模型

```
训练轮数: 5
批次大小: 16
学习率: 0.0001
```

### 2. 高质量训练大模型

```
训练轮数: 20
批次大小: 8
学习率: 0.00005
```

### 3. 快速微调适配领域

```
训练轮数: 3
学习率: 2e-5
勾选: 冻结Embedding层
```

### 4. 深度微调新任务

```
训练轮数: 10
学习率: 3e-5
不勾选: 冻结Embedding层
```

---

## 🔧 高级用法

### 通过CLI获得更多控制

如果需要更多参数控制，可以使用命令行：

```bash
# 训练
python -m apt_model train --epochs 20 --batch-size 8 --learning-rate 1e-4

# 微调
python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path finetune_data.txt \
  --epochs 5 \
  --learning-rate 1e-5 \
  --freeze-embeddings

# 聊天
python -m apt_model chat --model-path apt_model

# 评估
python -m apt_model evaluate --model-path apt_model
```

完整CLI文档请参考: [README.md](../../README.md)

---

## 📞 技术支持

- **完整文档**: [README.md](../../README.md)
- **微调指南**: [FINE_TUNING_GUIDE.md](../kernel/FINE_TUNING_GUIDE.md)
- **Optuna优化**: [OPTUNA_GUIDE.md](./OPTUNA_GUIDE.md)
- **问题反馈**: 在GitHub仓库创建Issue

---

## 📝 更新日志

### v1.0 (2025-12-01)
- ✨ 首次发布
- ✅ 支持训练、微调、聊天、评估
- ✅ 集成Optuna超参数优化
- ✅ 支持Windows、Linux、macOS
- ✅ 桌面快捷方式生成器

---

**Enjoy using APT Launcher! 🚀**
