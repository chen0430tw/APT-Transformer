# APT Model 快速启动指南

快速开始使用APT Model的WebUI和API服务。

---

## 🚀 快速启动

### 1. WebUI（推荐新手）

**基础启动**（无需认证，适合本地测试）：
```bash
python -m apt_model.webui.app --checkpoint-dir ./checkpoints
```

**生产启动**（带认证，推荐）：
```bash
python -m apt_model.webui.app \
  --checkpoint-dir ./checkpoints \
  --username admin \
  --password your_secure_password \
  --port 7860
```

**启动后访问**：
- 打开浏览器访问: http://localhost:7860
- 如果启用了认证，使用设置的用户名和密码登录

---

### 2. REST API

**基础启动**：
```bash
python -m apt_model.api.server --checkpoint-dir ./checkpoints
```

**自定义API密钥**：
```bash
python -m apt_model.api.server \
  --checkpoint-dir ./checkpoints \
  --api-key "your-secret-key-here" \
  --port 8000
```

**启动后访问**：
- API文档: http://localhost:8000/docs
- 替代文档: http://localhost:8000/redoc
- **重要**: 记下控制台显示的API密钥！

---

## 📋 启动时显示的信息

### WebUI启动示例

当您启动WebUI时，会看到类似这样的输出：

```
================================================================================
🚀 APT Model WebUI 启动中...
================================================================================

📋 配置信息:
  🌐 主机地址: 0.0.0.0
  🔌 端口: 7860
  📁 Checkpoint目录: ./checkpoints
  🌍 公共分享: ❌ 否
  🔐 访问控制: ✅ 已启用 (用户名: admin)

🌐 访问地址:
  📍 本地访问: http://localhost:7860
  📍 局域网访问: http://<你的IP>:7860

🔑 登录凭据:
  👤 用户名: admin
  🔒 密码: your_password

💡 功能说明:
  📊 训练监控 - 实时查看训练loss和学习率曲线
  🔍 梯度监控 - 监控梯度流和异常检测
  💾 Checkpoint管理 - 管理和加载模型检查点
  ✨ 推理测试 - 交互式文本生成

================================================================================
✅ WebUI 已启动！请在浏览器中打开上述地址
================================================================================
```

### API启动示例

当您启动API时，会看到类似这样的输出：

```
================================================================================
🚀 APT Model REST API 启动中...
================================================================================

📋 配置信息:
  🌐 主机地址: 0.0.0.0
  🔌 端口: 8000
  📁 Checkpoint目录: ./checkpoints
  🔄 热重载: ❌ 未启用
  🔐 PyTorch: ✅ 可用
  🚀 FastAPI: ✅ 可用

🌐 API访问地址:
  📍 本地访问: http://localhost:8000
  📍 局域网访问: http://<你的IP>:8000

📚 API文档:
  📖 Swagger UI: http://localhost:8000/docs
  📖 ReDoc: http://localhost:8000/redoc

🔑 API访问密钥 (自动生成):
  🔐 API Key: a7f3d9e2b8c1f4e6d5a9b3c7e2f8d1a4c9e5b7d3f1a8c4e6b2d9f5a1c7e3b8d4
  💡 请妥善保存此密钥，重启后将重新生成

💡 主要端点:
  🤖 推理服务:
     POST /api/generate - 单文本生成
     POST /api/batch_generate - 批量生成
  📊 训练监控:
     GET /api/training/status - 训练状态
     GET /api/training/gradients - 梯度数据
  💾 Checkpoint管理:
     GET /api/checkpoints - 列出checkpoints
     POST /api/checkpoints/load - 加载checkpoint

📝 使用示例:
  curl -X POST http://localhost:8000/api/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "你好", "max_length": 50}'

================================================================================
✅ API服务器已启动！
================================================================================
```

---

## 🔑 关于Token和密钥

### WebUI的用户名密码

**如何设置**：
```bash
python -m apt_model.webui.app --username admin --password secret123
```

**在哪里显示**：
- 启动时控制台会显示 🔑 登录凭据部分
- 包含用户名和密码（如上面的启动示例）

**如何使用**：
- 浏览器打开WebUI后，在登录页面输入用户名和密码
- 只需要登录一次（会话保持）

---

### API的访问密钥

**自动生成模式**（推荐开发）：
```bash
python -m apt_model.api.server --checkpoint-dir ./checkpoints
```
- 每次启动自动生成新的64字符密钥
- 密钥显示在控制台的 🔑 API访问密钥部分
- **重要**: 复制并保存这个密钥！

**自定义密钥模式**（推荐生产）：
```bash
python -m apt_model.api.server --api-key "my-permanent-key-12345"
```
- 使用固定的密钥，重启后不变
- 适合生产环境和自动化脚本

**密钥在哪里显示**：
- 启动时控制台会显示完整的API Key
- 自动生成的密钥会完整显示
- 自定义密钥会显示前16个字符（如：`my-permanent-ke...`）

**如何使用密钥**：
- 目前密钥存储在 `app.state.api_key` 中
- 未来版本会添加请求头验证：`Authorization: Bearer <your-api-key>`
- 当前版本主要用于记录和准备

---

## 🎯 查看启动演示

运行演示脚本查看启动效果（无需实际启动服务）：

```bash
python examples/demo_startup.py
```

这会展示WebUI和API启动时的完整控制台输出，让您了解实际启动时会看到什么。

---

## 💡 常见问题

### Q: 如何找到我的访问密钥/Token？

**WebUI**：
- 查看启动时控制台输出的 🔑 登录凭据部分
- 用户名和密码都会明文显示

**API**：
- 查看启动时控制台输出的 🔑 API访问密钥部分
- 自动生成的密钥会完整显示64个字符
- **务必复制保存！** 重启后会生成新密钥

### Q: Token/密钥丢失了怎么办？

**WebUI**：
- 如果忘记密码，需要重启服务并设置新密码
- 或者不使用 `--username/--password` 参数（无认证模式）

**API**：
- 自动生成模式：重启服务，会生成新密钥
- 自定义模式：使用 `--api-key` 参数设置固定密钥

### Q: 可以在浏览器中看到密钥吗？

**不能**。出于安全考虑：
- 密钥只在启动时的**控制台输出**中显示
- 浏览器界面不会显示密钥
- 请在启动时立即复制保存

### Q: 启动时没有看到彩色输出？

检查：
- 终端是否支持表情符号和颜色
- Windows用户可能需要使用Windows Terminal
- 或者使用 `--no-color` 参数（如果添加了此选项）

### Q: 如何局域网访问？

**找到你的IP地址**：
- Linux/Mac: `ifconfig` 或 `ip addr`
- Windows: `ipconfig`

**访问地址**：
- WebUI: `http://<你的IP>:7860`
- API: `http://<你的IP>:8000`

**防火墙设置**：
- 确保防火墙允许对应端口（7860或8000）
- Linux: `sudo ufw allow 7860`
- Windows: 在防火墙设置中添加端口例外

---

## 📚 更多文档

- **详细启动示例**: `examples/STARTUP_EXAMPLES.md`
- **完整使用指南**: `examples/USAGE_GUIDE.md`
- **演示脚本**: `python examples/demo_startup.py`

---

## 🆘 需要帮助？

查看帮助信息：

```bash
# WebUI帮助
python -m apt_model.webui.app --help

# API帮助
python -m apt_model.api.server --help
```

---

**提示**: 启动信息设计参考了生产级Bot系统的最佳实践，清晰展示所有必要的访问信息！
