# Scripts 目录

本目录包含所有项目相关的脚本和工具。

## 📁 目录结构

```
scripts/
├── README.md                      # 本文件
├── launchers/                     # 启动器相关
│   ├── APT_Launcher.bat          # Windows启动器
│   ├── APT_Launcher.sh           # Linux/Mac启动器
│   ├── apt_launcher.pyw          # GUI启动器
│   └── create_desktop_shortcut.py # 桌面快捷方式创建工具
├── archived/                      # 归档文件
│   └── APT_ALL_MODULES.tar.gz    # 旧的模块归档
├── run_best_training.sh          # 最佳训练参数运行
├── run_optuna_optimization.sh    # Optuna超参数优化
└── run_optuna_quick_test.sh      # Optuna快速测试
```

## 🚀 启动器 (launchers/)

### GUI启动器
```bash
# Windows
cd scripts/launchers && APT_Launcher.bat

# Linux/Mac
cd scripts/launchers && ./APT_Launcher.sh

# 跨平台GUI
python scripts/launchers/apt_launcher.pyw
```

### 创建桌面快捷方式
```bash
python scripts/launchers/create_desktop_shortcut.py
```

详细使用说明请参考: [启动器指南](../docs/product/LAUNCHER_README.md)

## 🎯 训练脚本

### 最佳参数训练
使用预设的最佳参数进行训练：
```bash
./scripts/run_best_training.sh
```

### Optuna超参数优化
完整的超参数搜索：
```bash
./scripts/run_optuna_optimization.sh
```

快速测试（少量试验）：
```bash
./scripts/run_optuna_quick_test.sh
```

详细使用说明请参考: [Optuna指南](../docs/product/OPTUNA_GUIDE.md)

## 📦 归档文件 (archived/)

存放历史归档文件，不影响当前项目运行。

## 💡 使用建议

### 新手用户
1. 使用GUI启动器快速开始
2. 运行最佳参数训练脚本

### 高级用户
1. 使用Optuna脚本优化超参数
2. 根据需求修改脚本参数

## 🔗 相关文档

- [启动器使用指南](../docs/product/LAUNCHER_README.md)
- [Optuna优化指南](../docs/product/OPTUNA_GUIDE.md)
- [微调指南](../docs/kernel/FINE_TUNING_GUIDE.md)
- [完整文档中心](../docs/README.md)

## 📝 注意事项

- 所有shell脚本需要执行权限: `chmod +x *.sh`
- Windows用户使用`.bat`文件
- GUI启动器需要安装tkinter: `pip install tk`
- 优化脚本需要安装optuna: `pip install optuna`
