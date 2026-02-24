# RunPod GPU 云平台使用指南

## 📌 概述

RunPod 是一个按秒计费的GPU云平台，提供高性能GPU实例（H200/H100/H800等），适合短期测试和训练任务。

**主要优势**:
- 按秒计费，用完即停
- SSH密钥认证，无需交互式登录
- 预装PyTorch/CUDA环境
- 支持自定义Docker镜像
- 快速部署（<2分钟）

---

## 🖥️ 硬件配置（已测试）

### NVIDIA H200 实例
- **显存**: 143GB
- **功耗**: 700W
- **价格**: ~$2-4/小时（按需付费）
- **CUDA**: 12.8
- **Driver**: 570.124.06

### 网络性能
- **上传速度**: ~24 MB/s
- **下载速度**: 1-3 MB/s
- **Ping延迟**: 6.9ms（到8.8.8.8）
- **Git clone**: 4秒完成1236个文件

---

## 🚀 快速开始

### Step 1: 创建RunPod实例

1. 访问 [RunPod](https://www.runpod.io/)
2. 注册账号并充值
3. **配置SSH密钥（必须）**
4. 选择GPU模板（推荐：NVIDIA H200）
5. 部署实例（2分钟内就绪）

#### SSH密钥配置（Windows + WSL）

**生成SSH密钥对**:
```bash
# 在Windows本地打开PowerShell或WSL
ssh-keygen -t rsa -b 4096 -C "runpod-key"

# 保存位置（默认）
# C:\Users\<用户名>\.ssh\id_rsa (私钥)
# C:\Users\<用户名>\.ssh\id_rsa.pub (公钥)
```

**添加公钥到RunPod**:
1. 复制公钥内容:
   ```bash
   # WSL
   cat ~/.ssh/id_rsa.pub

   # 或Windows PowerShell
   type $env:USERPROFILE\.ssh\id_rsa.pub
   ```

2. 打开RunPod控制台 → **Settings** → **SSH Keys**
3. 点击 **Add SSH Key**
4. 粘贴公钥内容并保存

**⚠️ 注意**:
- 私钥（`id_rsa`）保密，不要分享
- 公钥（`id_rsa.pub`）可以添加到多个平台
- 每个新实例需要使用相同的SSH密钥

### Step 2: 连接到实例

#### 查找连接信息

RunPod使用**NAT端口映射**，SSH端口不是标准的22端口。

1. 打开RunPod控制台 → **Pods**
2. 找到你的实例，点击 **Connect** 按钮
3. 复制SSH连接信息：
   ```
   SSH to connect: ssh root@<IP> -p <PORT>
   ```

**示例**:
```
IP: 103.196.86.60
Port: 57013 (不是22！)
```

#### 使用WSL连接（推荐）

```bash
wsl ssh root@103.196.86.60 -p 57013
```

#### 使用PowerShell连接

```powershell
ssh -o ConnectTimeout=10 root@103.196.86.60 -p 57013 "echo test"
```

**⚠️ PowerShell使用的SSH**:
- PowerShell默认使用 `C:\Program Files\Git\usr\bin\ssh.exe`（Git for Windows）
- 如果Git SSH有兼容性问题，可强制使用Windows OpenSSH：
  ```powershell
  C:\Windows\System32\OpenSSH\ssh.exe root@103.196.86.60 -p 57013
  ```

#### ❌ 密码登录不可用

**问题**: 尝试密码登录会卡住或超时

**原因**: RunPod **只支持SSH密钥认证**，不支持密码登录

**错误示例**:
```bash
# ❌ 错误：会提示输入密码，但输入任何密码都无法登录
ssh root@103.196.86.60 -p 57013
# root@103.196.86.60's password: （卡住）
```

**正确做法**:
```bash
# ✅ 正确：使用SSH密钥认证（自动）
wsl ssh root@103.196.86.60 -p 57013
# 直接登录，无需密码
```

#### ⚠️ 重要注意事项

1. **必须指定端口**:
   - RunPod使用NAT映射，SSH端口是动态分配的
   - 必须使用 `-p <PORT>` 指定端口
   - 不要使用默认的22端口

2. **必须配置SSH密钥**:
   - RunPod不支持密码登录
   - 必须先在控制台添加SSH公钥
   - 本地私钥必须在 `~/.ssh/id_rsa` 或使用 `-i` 指定

3. **必须使用WSL**:
   - WSL的SSH兼容性最好
   - Git Bash可能有问题（参考 MET0006）
   - PowerShell可能需要额外配置

4. **首次连接提示**:
   ```bash
   The authenticity of host '[103.196.86.60]:57013' can't be established.
   ED25519 key fingerprint is SHA256:xxxxx
   Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
   ```
   输入 `yes` 继续连接

### Step 3: 环境验证

```bash
# 检查GPU
nvidia-smi

# 检查Python
python3 --version

# 检查PyTorch
python3 -c "import torch; print(torch.__version__)"
```

**预装环境**:
- Python 3.12.3
- PyTorch 2.8.0+cu128
- CUDA 12.8
- transformers, datasets, tokenizers

### Step 4: 下载代码

```bash
# 克隆APT-Transformer仓库
cd /workspace
git clone https://github.com/chen0430tw/APT-Transformer.git
cd APT-Transformer
```

---

## 🧪 测试LECAC + Virtual VRAM

### 基础测试（10步）

```bash
python3 apt/trainops/scripts/pretrain_quickcook.py \
  --model-name tiny \
  --lecac-enable \
  --lecac-bits 2 \
  --lecac-alpha-warmup \
  --lecac-warmup-multiplier 5.0 \
  --virtual-vram-enable \
  --enable-nested-v16 \
  --max-steps 10 \
  --dataset-name C4 \
  --data-split 0.01
```

### 完整训练

```bash
python3 apt/trainops/scripts/pretrain_quickcook.py \
  --model-name tiny \
  --lecac-enable \
  --lecac-bits 2 \
  --lecac-alpha-warmup \
  --lecac-warmup-multiplier 5.0 \
  --virtual-vram-enable \
  --enable-nested-v16 \
  --max-steps 10000 \
  --dataset-name C4 \
  --batch-size 8 \
  --gradient-accumulation-steps 4
```

---

## 📊 网络测试

### 测试命令

```bash
# Ping测试
ping -c 3 8.8.8.8

# Git clone速度测试
cd /tmp
time git clone --depth 1 https://github.com/chen0430tw/APT-Transformer.git test_speed

# 上传速度测试（100MB）
dd if=/dev/zero bs=1M count=100 | curl -X POST -T - \
  https://speed.cloudflare.com/__up \
  -w '上传速度: %{speed_upload} bytes/sec\n'
```

### 实测结果

| 测试项 | 结果 |
|--------|------|
| Ping延迟 | 6.9ms |
| Git clone (1236文件) | 4秒 |
| 上传速度 | ~24 MB/s |
| 下载速度 | 1-3 MB/s |

---

## 💰 成本优化建议

### 短期测试
- ✅ 使用gtest等效分区（如果有）
- ✅ 设置合理的--max-steps限制
- ✅ 完成后立即停止实例

### 长期训练
- ⚠️ RunPod按小时计费，长时间训练成本高
- 💰 推荐使用晶创25/TWCC等固定价格集群

### 成本对比

| 平台 | H100价格 | H200价格 | 适用场景 |
|------|----------|----------|----------|
| RunPod | ~$2-3/小时 | ~$3-4/小时 | 短期测试（<1小时） |
| 晶创25 | 120元/GPU小时 | 150元/GPU小时 | 长期训练（>10小时） |
| TWCC | - | - | 中等规模（V100 32GB） |

---

## 🔧 常见问题

### SSH连接超时
**问题**: `ssh: connect to host ... port ...: Connection timed out`

**原因**: 实例被删除或网络问题

**解决**:
1. 检查RunPod控制台，确认实例状态
2. 如果实例被删除，重新创建
3. 避免长时间无操作（RunPod会自动关闭空闲实例）

### 下载速度慢
**问题**: pip install 或数据下载很慢

**解决**:
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package-name

# 使用--quiet参数减少输出
pip install --quiet package-name
```

### 实例被自动删除
**原因**: 余额不足或超时

**解决**:
1. 检查账户余额
2. 设置合理的自动停止时间
3. 重要数据及时保存到本地

---

## 📝 最佳实践

### 1. 快速验证流程
```
本地WSL测试 (免费) → RunPod快速验证 (付费但快) → 集群正式训练 (稳定)
```

### 2. 避免浪费
- ❌ 不要在RunPod上运行长时间数据下载
- ❌ 不要频繁查询队列状态
- ✅ 数据预处理提前完成
- ✅ 使用--max-steps限制测试步数

### 3. 监控训练
```bash
# 后台运行训练
nohup python3 -u script.py > train.log 2>&1 &

# 实时查看日志
tail -f train.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

---

## 🔗 相关链接

- [RunPod官网](https://www.runpod.io/)
- [RunPod文档](https://docs.runpod.io/)
- [APT-Transformer仓库](https://github.com/chen0430tw/APT-Transformer)
- [LECaC Alpha Warmup文档](../QUICKSTART_WARMUP.md)
- [NaN调试清单](../DEBUG_NAN_CHECKLIST.md)

---

## ⚠️ 效率管理教训

### 问题：拖拖拉拉导致实例被关闭

**症状**:
- RunPod实例按秒计费（~$0.001/秒）
- 调试过程中效率低下，反复测试
- 最终实例因超时或余额不足被自动关闭
- 浪费了大量时间和金钱

**低效行为示例**:

1. **反复测试相同配置**
   - ❌ 测试 multiplier 3.0 失败
   - ❌ 测试 multiplier 1.5 失败
   - ❌ 测试 multiplier 5.0 失败
   - ✅ 应该：一次anomaly detection定位问题

2. **没有使用正确的调试工具**
   - ❌ 盯着loss值猜测问题
   - ❌ 反复修改参数看是否解决
   - ✅ 应该：第一时间使用`torch.autograd.detect_anomaly()`

3. **测试周期过长**
   - ❌ 每次测试等几分钟看结果
   - ❌ 没有并行测试多个配置
   - ✅ 应该：本地快速验证后再上云测试

4. **实例时间管理失控**
   - ❌ 实例开着没做正事（240分钟测试周期）
   - ❌ 等待时没有及时停止实例
   - ✅ 应该：设置明确的测试时间表

### 经验教训

**1. 本地优先原则**
```
本地WSL测试（免费）→ 快速验证代码逻辑 → 云端验证（付费）
```

- 不要在云端做本地可以做的测试
- 使用小模型快速验证修复方向
- 确认修复有效后再上云

**2. 诊断工具优先**
- 第一时间使用`torch.autograd.detect_anomaly()`
- 第一时间使用`pdb`或`ipdb`断点调试
- 不要靠猜测和试错

**3. 并行测试**
```bash
# ❌ 串行测试（浪费6分钟）
测试配置1 → 等2分钟 → 失败
测试配置2 → 等2分钟 → 失败
测试配置3 → 等2分钟 → 失败

# ✅ 并行测试（节省4分钟）
同时启动配置1/2/3 → 一起等2分钟 → 一起看结果
```

**4. 时间预算管理**
```bash
# 测试前制定时间表
0-5分钟:   上传代码
5-10分钟:  快速验证（5步测试）
10-15分钟: 完整验证（30步测试）
15-20分钟: 清理和停止实例
```

**5. 快速失败原则**
- 如果5步内就NaN，立即停止，不要等100步
- 如果anomaly detection报错，立即停止分析
- 如果代码修改没逻辑，不要盲目测试

### 实际案例

**本次调试的时间浪费**:
- 测试1（multiplier 3.0）: 8分钟 → 失败
- 测试2（multiplier 1.5）: 8分钟 → 失败
- 测试3（anomaly detection）: 10分钟 → 找到原因
- 测试4（multiplier 5.0）: 8分钟 → 失败
- 测试5（fix验证）: 8分钟 → 成功
- **总计**: 42分钟 = $2-4 USD

**正确做法**:
```
0-5分钟:   本地验证代码语法
5-10分钟:  小模型云端测试（5步）
10-15分钟: anomaly detection定位问题
15-20分钟: 实施fix并验证
20-25分钟: 停止实例
总计: 25分钟 = $1-2 USD（节省40%时间+50%成本）
```

### 最佳实践总结

1. **本地测试优先**: 能本地不云端
2. **工具优先**: 先诊断后修复
3. **小步快跑**: 5步测试代替100步
4. **并行优于串行**: 多配置同时测
5. **严格时间控制**: 设置闹钟，到点停止
6. **及时止损**: 发现问题立即分析，不要盲目继续

---

## 🐛 NaN问题技术记录

### 问题描述

在RunPod H200上测试完整模型(vocab=65536)时，出现训练NaN问题：

**配置**:
- 模型: apt (d_model=768, 12 layers)
- LECaC: INT2量化 + Alpha warmup (multiplier 3.0)
- Virtual VRAM: nested v16启用
- 批次大小: 16

**症状**:
- Step 0: Loss=11.24 ✅
- Step 1: ❌ Loss=NaN

### 根本原因

使用`torch.autograd.detect_anomaly()`捕获到：

```python
RuntimeError: Function 'ScaledDotProductEfficientAttentionBackward0'
returned nan values in its 0th output.
```

**问题**: Virtual VRAM的pack_hook使用了`t.detach()`切断了梯度链，导致大模型Attention backward产生NaN。

### 修复方案

`virtual_vram.py:871-876`: 对于`requires_grad=True`的tensor，直接返回原tensor，不进行offload。

**效果**: ✅ 完整模型训练正常
**代价**: ❌ 训练时Virtual VRAM基本失效

### 正确实现方向（待完成）

需要实现兼容autograd的量化机制，让量化的tensor也能正确反向传播梯度。

---

## 📅 更新日志

- **2026-02-24**: 初始版本，基于H200实例测试
- 测试配置: H200 143GB, PyTorch 2.8.0, CUDA 12.8
- 验证功能: LECaC INT2 + Virtual VRAM v16 + Alpha Warmup
- **2026-02-24**: 添加NaN问题调试记录和效率管理教训
