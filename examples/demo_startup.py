#!/usr/bin/env python3
"""
演示脚本：展示WebUI和API的启动信息

不实际启动服务，只展示启动时的控制台输出
"""

def demo_webui_startup():
    """演示WebUI启动输出"""
    print("\n" + "=" * 80)
    print("🌐 WebUI 启动演示")
    print("=" * 80)
    print()
    print("命令: python -m apt_model.webui.app --checkpoint-dir ./checkpoints --username admin --password secret")
    print()

    # 模拟启动输出
    print("=" * 80)
    print("🚀 APT Model WebUI 启动中...")
    print("=" * 80)
    print()

    print("📋 配置信息:")
    print("  🌐 主机地址: 0.0.0.0")
    print("  🔌 端口: 7860")
    print("  📁 Checkpoint目录: ./checkpoints")
    print("  🌍 公共分享: ❌ 否")
    print("  🔐 访问控制: ✅ 已启用 (用户名: admin)")
    print()

    print("🌐 访问地址:")
    print("  📍 本地访问: http://localhost:7860")
    print("  📍 局域网访问: http://<你的IP>:7860")
    print()

    print("🔑 登录凭据:")
    print("  👤 用户名: admin")
    print("  🔒 密码: secret")
    print()

    print("💡 功能说明:")
    print("  📊 训练监控 - 实时查看训练loss和学习率曲线")
    print("  🔍 梯度监控 - 监控梯度流和异常检测")
    print("  💾 Checkpoint管理 - 管理和加载模型检查点")
    print("  ✨ 推理测试 - 交互式文本生成")
    print()

    print("=" * 80)
    print("✅ WebUI 已启动！请在浏览器中打开上述地址")
    print("=" * 80)
    print()


def demo_api_startup():
    """演示API启动输出"""
    print("\n" + "=" * 80)
    print("🚀 API 启动演示")
    print("=" * 80)
    print()
    print("命令: python -m apt_model.api.server --checkpoint-dir ./checkpoints")
    print()

    # 模拟启动输出
    print("=" * 80)
    print("🚀 APT Model REST API 启动中...")
    print("=" * 80)
    print()

    print("📋 配置信息:")
    print("  🌐 主机地址: 0.0.0.0")
    print("  🔌 端口: 8000")
    print("  📁 Checkpoint目录: ./checkpoints")
    print("  🔄 热重载: ❌ 未启用")
    print("  🔐 PyTorch: ⚠️  不可用 (演示模式)")
    print("  🚀 FastAPI: ⚠️  不可用 (演示模式)")
    print()

    print("🌐 API访问地址:")
    print("  📍 本地访问: http://localhost:8000")
    print("  📍 局域网访问: http://<你的IP>:8000")
    print()

    print("📚 API文档:")
    print("  📖 Swagger UI: http://localhost:8000/docs")
    print("  📖 ReDoc: http://localhost:8000/redoc")
    print()

    print("🔑 API访问密钥 (自动生成):")
    print("  🔐 API Key: a7f3d9e2b8c1f4e6d5a9b3c7e2f8d1a4c9e5b7d3f1a8c4e6b2d9f5a1c7e3b8d4")
    print("  💡 请妥善保存此密钥，重启后将重新生成")
    print()

    print("💡 主要端点:")
    print("  🤖 推理服务:")
    print("     POST /api/generate - 单文本生成")
    print("     POST /api/batch_generate - 批量生成")
    print("  📊 训练监控:")
    print("     GET /api/training/status - 训练状态")
    print("     GET /api/training/gradients - 梯度数据")
    print("  💾 Checkpoint管理:")
    print("     GET /api/checkpoints - 列出checkpoints")
    print("     POST /api/checkpoints/load - 加载checkpoint")
    print()

    print("📝 使用示例:")
    print("  curl -X POST http://localhost:8000/api/generate \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"text": "你好", "max_length": 50}\'')
    print()

    print("=" * 80)
    print("✅ API服务器已启动！")
    print("=" * 80)
    print()


def main():
    """主函数"""
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "APT Model 启动信息演示" + " " * 36 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("这个脚本演示WebUI和API启动时的控制台输出")
    print("实际使用时，启动信息会自动显示")
    print()

    # 演示WebUI
    demo_webui_startup()

    print("\n" + "─" * 80 + "\n")

    # 演示API
    demo_api_startup()

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 30 + "演示完成" + " " * 40 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("💡 提示:")
    print("  - WebUI启动: python -m apt_model.webui.app --help")
    print("  - API启动:   python -m apt_model.api.server --help")
    print("  - 详细文档: examples/STARTUP_EXAMPLES.md")
    print()


if __name__ == '__main__':
    main()
