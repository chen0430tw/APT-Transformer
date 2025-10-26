#!/bin/bash
# APT插件安装脚本
# 用途: 快速安装所有插件依赖

set -e  # 遇到错误立即退出

echo "🚀 APT插件依赖安装脚本"
echo "======================================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3"
    exit 1
fi

echo "✅ Python版本: $(python3 --version)"
echo ""

# 第一优先级:外部集成插件
echo "📦 第一优先级:外部集成插件"
echo "======================================"

echo "  [1/4] 安装 HuggingFace 相关包..."
pip install transformers datasets huggingface_hub --break-system-packages -q
echo "  ✅ HuggingFace包安装完成"

echo "  [2/4] 安装 AWS S3 支持..."
pip install boto3 --break-system-packages -q
echo "  ✅ boto3安装完成"

echo "  [3/4] 安装 阿里云 OSS 支持..."
pip install oss2 --break-system-packages -q
echo "  ✅ oss2安装完成"

echo "  [4/4] 安装 ModelScope 支持..."
pip install modelscope --break-system-packages -q || echo "  ⚠️  ModelScope安装失败(可选)"

echo ""

# 第二优先级:高级训练插件(可选)
echo "📦 第二优先级:高级训练插件(可选)"
echo "======================================"

read -p "是否安装高级剪枝库 torch-pruning? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  安装 torch-pruning..."
    pip install torch-pruning --break-system-packages -q || echo "  ⚠️  torch-pruning安装失败(可选)"
    echo "  ✅ torch-pruning安装完成"
else
    echo "  跳过 torch-pruning"
fi

echo ""

# 第三优先级:工具类插件(可选)
echo "📦 第三优先级:工具类插件(可选)"
echo "======================================"

read -p "是否安装调试工具(wandb, tensorboard)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  安装调试工具..."
    pip install wandb tensorboard --break-system-packages -q
    echo "  ✅ 调试工具安装完成"
else
    echo "  跳过调试工具"
fi

echo ""

read -p "是否安装数据处理工具(pandas, openpyxl)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  安装数据处理工具..."
    pip install pandas openpyxl beautifulsoup4 --break-system-packages -q
    echo "  ✅ 数据处理工具安装完成"
else
    echo "  跳过数据处理工具"
fi

echo ""
echo "======================================"
echo "✅ APT插件依赖安装完成!"
echo "======================================"
echo ""
echo "📝 下一步:"
echo "  1. 将插件文件复制到 apt_model/plugins/ 目录"
echo "  2. 在 plugin_system.py 中注册插件"
echo "  3. 更新配置文件启用所需插件"
echo "  4. 运行示例代码测试插件功能"
echo ""
echo "📚 详细文档请查看: README.md 和 APT_Plugin_Implementation_Plan.md"
echo ""
