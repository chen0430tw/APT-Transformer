# APT APX Package Format

APT Package Exchange - 模型打包和分发标准

## 概述

`apt.apx` 定义了APT模型的标准打包格式，用于模型分发、版本管理和部署。

APX = **APT Package Exchange**

## 为什么需要APX？

在AI模型生态中，我们需要：
- **统一格式** - 标准化的模型打包格式
- **版本管理** - 清晰的版本控制和依赖管理
- **安全性** - 数字签名和完整性验证
- **可移植性** - 跨平台、跨环境部署
- **元数据** - 完整的模型描述和使用说明

APX提供完整的解决方案。

## 目录结构

```
apt/apx/
├── packaging/     # 模型打包工具
├── distribution/  # 分发和部署
└── validation/    # 包验证和签名
```

## APX包格式

### 包结构

```
model_name-1.0.0.apx
├── manifest.json         # 包清单
├── metadata.yaml         # 模型元数据
├── model/               # 模型文件
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer/
├── checkpoints/         # 检查点（可选）
├── artifacts/           # 训练产物（可选）
├── docs/               # 文档
│   ├── README.md
│   └── MODEL_CARD.md
├── examples/           # 示例代码
└── signature.sig       # 数字签名
```

### manifest.json

```json
{
  "name": "apt-large-v1",
  "version": "1.0.0",
  "description": "APT Large Model v1.0",
  "author": "APT Team",
  "license": "MIT",
  "created_at": "2026-01-22T10:00:00Z",
  "apt_version": "2.0.0",
  "dependencies": {
    "torch": ">=2.0.0",
    "transformers": ">=4.30.0"
  },
  "files": [
    {
      "path": "model/pytorch_model.bin",
      "size": 5368709120,
      "checksum": "sha256:abc123..."
    }
  ],
  "signature": {
    "algorithm": "RSA-SHA256",
    "key_id": "apt-team-key-2026"
  }
}
```

### metadata.yaml

```yaml
model:
  name: apt-large-v1
  architecture: apt_large
  parameters: 1.5B
  hidden_size: 2048
  num_layers: 32
  num_attention_heads: 32

training:
  dataset: "APT Corpus 2026"
  training_steps: 500000
  training_time: "200 GPU-days"
  hardware: "64x A100 80GB"

performance:
  perplexity: 12.3
  accuracy: 0.95
  throughput: "1000 tokens/sec"

capabilities:
  - text-generation
  - question-answering
  - summarization
  - multimodal

languages:
  - zh-CN
  - en-US

tags:
  - transformer
  - autopoietic
  - large-language-model
```

## 模块说明

### 1. packaging/

模型打包工具：

```python
from apt.apx.packaging import package_model

# 打包模型
package_model(
    model_path='checkpoints/model-final/',
    output_path='apt-large-v1.apx',
    metadata={
        'name': 'apt-large-v1',
        'version': '1.0.0',
        'description': 'APT Large Model'
    },
    sign=True  # 数字签名
)
```

功能：
- 模型打包
- 资源打包
- 元数据生成
- 压缩优化

### 2. distribution/

分发和部署：

```python
from apt.apx.distribution import publish_package, download_package

# 发布到仓库
publish_package(
    package='apt-large-v1.apx',
    repository='https://models.apt-transformer.org',
    visibility='public'
)

# 从仓库下载
package = download_package(
    name='apt-large-v1',
    version='1.0.0',
    destination='models/'
)
```

功能：
- 模型发布
- 版本管理
- 下载工具
- 部署辅助

### 3. validation/

包验证和签名：

```python
from apt.apx.validation import validate_package, verify_signature

# 验证包完整性
is_valid = validate_package('apt-large-v1.apx')

# 验证数字签名
is_signed = verify_signature(
    package='apt-large-v1.apx',
    public_key='apt-team-public-key.pem'
)
```

功能：
- 完整性检查
- 数字签名验证
- 安全扫描
- 依赖验证

## 使用示例

### 打包模型

```python
from apt.apx.packaging import APXPackager

# 创建打包器
packager = APXPackager()

# 添加模型文件
packager.add_model('checkpoints/model-final/')

# 添加元数据
packager.set_metadata({
    'name': 'my-custom-apt',
    'version': '1.0.0',
    'description': 'My custom APT model',
    'author': 'Your Name',
    'license': 'MIT'
})

# 添加文档
packager.add_docs('docs/')

# 添加示例
packager.add_examples('examples/')

# 打包并签名
packager.build(
    output='my-custom-apt-1.0.0.apx',
    sign=True,
    private_key='my-private-key.pem'
)
```

### 发布模型

```python
from apt.apx.distribution import APXPublisher

# 创建发布器
publisher = APXPublisher(
    repository='https://my-model-hub.com',
    api_key='your-api-key'
)

# 上传包
publisher.publish(
    package='my-custom-apt-1.0.0.apx',
    visibility='public',
    tags=['transformer', 'chinese', 'custom']
)

print(f"Published: {publisher.get_url()}")
# https://my-model-hub.com/models/my-custom-apt/1.0.0
```

### 下载和使用模型

```python
from apt.apx.distribution import download_and_load

# 下载并加载模型
model = download_and_load(
    name='apt-large-v1',
    version='1.0.0',
    cache_dir='~/.apt/models/'
)

# 直接使用
output = model.generate("你好世界")
```

### 验证模型包

```python
from apt.apx.validation import APXValidator

# 创建验证器
validator = APXValidator()

# 验证包
result = validator.validate('apt-large-v1.apx')

if result.is_valid:
    print("✓ Package is valid")
    print(f"  Files: {result.num_files}")
    print(f"  Size: {result.total_size}")
    print(f"  Signature: {result.signature_valid}")
else:
    print("✗ Package validation failed:")
    for error in result.errors:
        print(f"  - {error}")
```

## CLI工具

APX提供命令行工具：

```bash
# 打包模型
apt-apx pack \
  --model checkpoints/model-final/ \
  --output apt-large-v1.apx \
  --metadata metadata.yaml \
  --sign

# 发布模型
apt-apx publish \
  --package apt-large-v1.apx \
  --repository https://models.apt-transformer.org \
  --visibility public

# 下载模型
apt-apx download \
  --name apt-large-v1 \
  --version 1.0.0 \
  --output models/

# 验证模型
apt-apx validate apt-large-v1.apx

# 解包模型
apt-apx unpack apt-large-v1.apx --output unpacked/

# 查看信息
apt-apx info apt-large-v1.apx

# 列出文件
apt-apx list apt-large-v1.apx
```

## 版本管理

APX使用语义化版本（Semantic Versioning）：

```
major.minor.patch[-prerelease][+build]

例如：
- 1.0.0        # 稳定版本
- 1.0.1        # 补丁版本
- 1.1.0        # 小版本升级
- 2.0.0        # 大版本升级
- 1.0.0-alpha  # Alpha版本
- 1.0.0-beta   # Beta版本
- 1.0.0-rc.1   # Release Candidate
```

版本兼容性：
- **Patch (x.y.Z)** - 向后兼容的bug修复
- **Minor (x.Y.z)** - 向后兼容的功能增加
- **Major (X.y.z)** - 不兼容的API变更

## 模型仓库

### 官方仓库

```
https://models.apt-transformer.org/
├── apt-small/
│   ├── 1.0.0/
│   ├── 1.1.0/
│   └── 2.0.0/
├── apt-base/
│   └── 1.0.0/
├── apt-large/
│   └── 1.0.0/
└── ...
```

### 私有仓库

搭建私有APX仓库：

```bash
# 使用Docker部署
docker run -d \
  -p 8080:8080 \
  -v /data/models:/models \
  apt-registry:latest

# 配置客户端
apt-apx config set repository https://my-registry.com
apt-apx config set api-key YOUR_API_KEY
```

## 安全性

### 数字签名

APX使用RSA或Ed25519签名：

```python
from apt.apx.validation import sign_package

# 签名包
sign_package(
    package='model.apx',
    private_key='private-key.pem',
    algorithm='RSA-SHA256'
)
```

### 完整性验证

每个文件都有SHA-256校验和：

```json
{
  "files": [
    {
      "path": "model/pytorch_model.bin",
      "checksum": "sha256:abc123..."
    }
  ]
}
```

### 安全扫描

APX可以扫描包中的安全问题：

```bash
apt-apx scan model.apx --check malware --check vulnerabilities
```

## 配置文件

APX配置（`~/.apt/apx.yaml`）：

```yaml
repositories:
  - name: official
    url: https://models.apt-transformer.org
    priority: 1

  - name: private
    url: https://my-registry.com
    api_key: ${APX_API_KEY}
    priority: 2

cache:
  directory: ~/.apt/models/
  max_size: 100GB
  ttl: 7d

security:
  verify_signatures: true
  allow_unsigned: false
  trusted_keys:
    - apt-team-public-key.pem

download:
  parallel_downloads: 4
  resume: true
  timeout: 300
```

## 迁移状态

🚧 **当前状态**: Skeleton已创建，内容将在PR-5中实现

实现计划：
- [ ] PR-5: 实现打包工具
- [ ] PR-5: 实现分发系统
- [ ] PR-5: 实现验证和签名
- [ ] PR-5: 开发CLI工具
- [ ] PR-5: 搭建官方仓库

## 与其他格式的比较

| 特性 | APX | HuggingFace | ONNX | TorchScript |
|-----|-----|------------|------|-------------|
| 格式类型 | 完整包 | Hub托管 | 模型格式 | 模型格式 |
| 元数据 | ✅ 丰富 | ✅ 丰富 | ⚠️ 有限 | ⚠️ 有限 |
| 签名 | ✅ | ❌ | ❌ | ❌ |
| 版本管理 | ✅ | ✅ | ⚠️ | ⚠️ |
| 自托管 | ✅ | ⚠️ | N/A | N/A |
| 跨框架 | ⚠️ PyTorch | ✅ | ✅ | ❌ |

APX优势：
- 🔐 内置安全机制（签名、校验）
- 📦 完整打包（模型+文档+示例）
- 🏢 支持私有部署
- 📝 丰富的元数据

## 最佳实践

1. **始终签名发布的模型** - 确保模型来源可信
2. **详细的元数据** - 提供完整的模型信息
3. **语义化版本** - 遵循版本规范
4. **包含文档** - README和MODEL_CARD
5. **提供示例** - 展示如何使用模型

## 示例：完整工作流

```python
from apt.trainops.engine import Trainer
from apt.apx.packaging import APXPackager
from apt.apx.distribution import APXPublisher

# 1. 训练模型
trainer = Trainer(model=model, ...)
trainer.train()
trainer.save_model('checkpoints/final/')

# 2. 打包模型
packager = APXPackager()
packager.add_model('checkpoints/final/')
packager.set_metadata({
    'name': 'my-apt-model',
    'version': '1.0.0',
    'description': 'My trained APT model'
})
packager.add_docs('docs/')
packager.build('my-apt-model-1.0.0.apx', sign=True)

# 3. 发布模型
publisher = APXPublisher(repository='https://models.apt-transformer.org')
publisher.publish('my-apt-model-1.0.0.apx', visibility='public')

print(f"✓ Model published: {publisher.get_url()}")
```

## API文档

详细API文档：https://apt-transformer.readthedocs.io/apx/

## 测试

```bash
# 测试APX模块
pytest apt/apx/tests/

# 测试打包
pytest apt/apx/tests/test_packaging.py

# 测试签名
pytest apt/apx/tests/test_validation.py
```

## 相关链接

- [Model Domain](../model/README.md) - 模型域
- [TrainOps Domain](../trainops/README.md) - 训练域
- [APX Specification](../../docs/specs/apx_format.md)
- [Model Hub Guide](../../docs/guides/model_hub.md)

---

**Version**: 2.0.0-alpha
**Status**: Skeleton (内容实现中)
**Last Updated**: 2026-01-22
**Specification**: APX Format v1.0
