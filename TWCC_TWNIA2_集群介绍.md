# 台湾杉二号 (TWNIA2) HPC集群完整介绍

> **文档版本**: v1.0
> **更新日期**: 2026-02-23
> **集群**: 台湾杉二号 (Taiwania 2)
> **所属**: 国家实验研究院
> **登录地址**: ln01.twcc.ai
> **别名**: twcc (SSH配置)

---

## 📑 目录

1. [快速概览](#快速概览)
2. [硬件配置](#硬件配置)
3. [网络架构](#网络架构)
4. [存储系统](#存储系统)
5. [软件环境](#软件环境)
6. [Slurm作业调度](#slurm作业调度)
7. [账户与权限](#账户与权限)
8. [使用规则与注意事项](#使用规则与注意事项)
9. [费用优化建议](#费用优化建议)
10. [常用命令](#常用命令)
11. [快速上手](#快速上手)
12. [与晶创25对比](#与晶创25对比)

---

## 🚀 快速概览

### 基本信息
- **集群名称**: 台湾杉二号 (Taiwania 2)
- **管理机构**: 国家实验研究院
- **登录地址**: `ln01.twcc.ai`
- **操作系统**: CentOS 8.4
- **调度系统**: Slurm
- **网络**: InfiniBand EDR 100Gbps + NVLINK 300GB/s

### 规模统计
```
┌─────────────────────────────────────────────┐
│ 节点总数: 252个计算节点                    │
│ GPU总数:   2,016片 (V100 32GB)            │
│ CPU总数:   9,216核 (252×36核)             │
│ 内存总量:   193 TB (252×768GB)            │
│ 峰值性能:   9 PFLOPS                       │
└─────────────────────────────────────────────┘
```

### Queue 划分
| Queue | 最长时长 | 高优先权 | Job上限 | GPU上限 | 适用场景 |
|-------|---------|---------|---------|---------|---------|
| **gtest** | 0.5小时 | | 5 | 40 | 快速测试调试 |
| **gp1d** | 24小时 | | 20 | 40 | 短期训练 |
| **gp2d** | 48小时 | | 20 | 40 | 中期训练 |
| **gp4d** | 96小时 | | 20 | 40 | 长期训练 |
| **express** | 96小时 | ✓ | 20 | 256 | 企业高优先权 |

**限制说明**:
- 用户最多提交 20 个计算工作
- 所有计算工作加总最多使用 40 张 GPUs (express队列256 GPUs)
- 每个计算工作至少指定 1 张 GPU
- **资源比例**: 1 GPU : 4 CPU : 90 GB Memory

---

## 💻 硬件配置

### 计算节点规格
```
节点数量: 252个
每个节点:
  ├─ CPU: Intel Xeon Gold 6154 @ 3.0GHz (18核×2插槽 = 36核)
  ├─ 内存: 768 GB DDR4 (24×32GB)
  ├─ GPU: 8× NVIDIA Tesla V100-SXM2 32GB
  ├─ 存储: 4TB NVMe (数据暂存)
  ├─ 网络: 4× Mellanox InfiniBand EDR 100Gb
  └─ 架构: x86_64

GPU互联:
  ├─ NVLINK: 最高 300 GB/s GPU间通信
  └─ IB网络: 100 Gbps节点间通信
```

### 性能指标
```
峰值性能: 9 PFLOPS (每秒9千兆次浮点运算)
GPU型号: Tesla V100-SXM2 32GB HBM2
显存带宽: 900 GB/s
CUDA核心: 5120个
Tensor核心: 640个 (FP16/BF16加速)
```

---

## 🌐 网络架构

### 实测网络性能
```
登录节点 (ln01.twcc.ai):
  CPU: 72核 (2×18核×2线程)
  GPU: 8× V100 32GB (登录节点可用，但限制使用)
  内存: 378GB

网络延迟:
  HiNet DNS: 2.6ms
  Google: 6.9ms
  台湾ISP: 2-7ms
  丢包率: 0%

带宽测试:
  下载速度: 2.3 MB/s (18.4 Mbps)
  上传速度: 1.0 MB/s (7.8 Mbps)
  磁盘写入: 2.4-2.6 GB/s (本地NVMe)
```

### ⚠️ 登录节点限制
```
GPU使用限制:
  └─ GPU进程运行超过5分钟 → 自动清除所有进程

CPU使用限制:
  └─ CPU使用率超过400% → 自动终止该进程

正确做法:
  ✅ 使用 sbatch 提交计算任务
  ✅ 使用 salloc 分配交互式计算节点
  ❌ 在登录节点直接运行训练脚本
```

---

## 💾 存储系统

### 文件系统概览
```
/home (HFS 高速文件系统):
  容量: 1.8 PB
  已用: 955 TB (54%)
  用户配额: 100 GB (免费)
  位置: /home/twsuday816
  用途: 个人文件、配置、SSH密钥
  特点: 长期存储、跨节点共享

/work (HFS 高速文件系统):
  容量: 7.6 PB
  已用: 5.5 PB (72%)
  用户配额: 100 GB (免费)
  位置: /work/twsuday816
  用途: 训练脚本、作业输出、工作数据
  特点: 高IOPS、跨节点共享、定期清理
```

### 目录使用建议
```
推荐结构:

/work/twsuday816/
├── scripts/      # 训练脚本
├── data/        # 输入数据集 (注意100GB限制)
├── models/      # 保存的模型
├── results/     # 输出结果
└── logs/        # 日志文件

/home/twsuday816/
├── .ssh/        # SSH密钥
├── .local/      # Python包安装位置
└── .config/     # 配置文件
```

### ⚠️ 存储限制与建议
```
配额情况:
  /home: 100 GB (长期存储)
  /work: 100 GB (暂存，定期清理)

重要提醒:
  ❌ /work 空间会定期自动清理，不适合长期存储
  ❌ 100GB配额较小，不适合大规模数据集
  ✅ 训练数据建议使用外部存储或按需下载
  ✅ 重要结果及时下载到本地或备份

与晶创25对比:
  晶创25: /home 501TB + /work 3.6PB (几乎无限)
  TWNIA2: /home 100GB + /work 100GB (严格限制)
```

---

## 🔧 软件环境

### 模块系统 (Lmod)
```
可用模块类别:

CUDA (默认cuda/12.8):
  cuda/10.2, cuda/11.4, cuda/11.7, cuda/12.3, cuda/12.8(默认)

OpenMPI:
  openmpi/4.1.6_ucx1.14.1_cuda12.3
  openmpi/5.0.2_ucx1.14.1_cuda12.3 (默认)

编译器:
  gcc7/7.3.1, gcc8/8.3.1, gcc9/9.3.1, gcc10/10.2.1
  intel/2018, intel/2020 (默认)

容器:
  singularity (已安装)

Python:
  miniconda3/conda24.5.0_py3.9
  miniforge/24.7.1-2

其他工具:
  cmake/3.23.2, ffmpeg/7.1, git/2.46.2, s5cmd/2.2.2
```

### 当前环境 (ln01)
```
用户: twsuday816
组: TRI900137, ENT114035, ENT902239, slurm, chem, lammps, qchem, cmp, gaussian16

工作目录:
  /work/twsuday816 (已有其他项目: grabio_simluation, fmri.py等)
```

---

## ⚙️ Slurm作业调度

### 资源分配规则
```bash
基本单位:
  1 GPU : 4 CPU : 90 GB Memory

示例:
  └─ 1 GPU  → 4 CPU + 90GB内存
  └─ 2 GPU  → 8 CPU + 180GB内存
  └─ 4 GPU  → 16 CPU + 360GB内存
  └─ 8 GPU  → 36 CPU + 768GB内存 (整节点)
```

### 资源限制总结
```bash
用户限制:
  MaxJobCount: 20个job (gtest: 5个)
  MaxGPUUsage: 40 GPUs (express: 256 GPUs)
  MaxTime: 按 Queue (0.5h - 96h)

节点配置:
  GPUs/节点: 8
  CPUs/节点: 36
  内存/节点: 768 GB
  暂存盘/节点: 4TB NVMe
```

### 常用Slurm脚本模板

#### 1. 快速测试 (gtest Queue)
```bash
#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=gtest
#SBATCH --account=ENT114035
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/work/twsuday816/test_%j.out
#SBATCH --error=/work/twsuday816/test_%j.err

module load cuda/12.8
module load miniconda3/conda24.5.0_py3.9

# 你的训练脚本
python train.py
```

#### 2. 单节点训练 (gp1d Queue)
```bash
#!/bin/bash
#SBATCH --job-name=training_1d
#SBATCH --partition=gp1d
#SBATCH --account=ENT114035
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/work/twsuday816/train_%j.out
#SBATCH --error=/work/twsuday816/train_%j.err

module load cuda/12.8
module load miniconda3/conda24.5.0_py3.9

srun python train.py --batch-size 128 --gpus 4
```

#### 3. 多节点训练 (gp4d Queue)
```bash
#!/bin/bash
#SBATCH --job-name=training_4d
#SBATCH --partition=gp4d
#SBATCH --account=ENT114035
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --output=/work/twsuday816/train_%j.out
#SBATCH --error=/work/twsuday816/train_%j.err

module load cuda/12.8
module load openmpi/5.0.2_ucx1.14.1_cuda12.3
module load miniconda3/conda24.5.0_py3.9

# 分布式训练
srun python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    train.py
```

---

## 👤 账户与权限

### 账户信息
```bash
用户名: twsuday816
UID: 204634

关联账户:
  ├─ ENT114035 (企业账户) - GID: 102622
  ├─ TRI900137 (国研院账户) - GID: 102640
  └─ ENT902239 (另一账户)

所属组:
  TRI900137, ENT114035, ENT902239, slurm,
  chem, lammps, qchem, cmp, gaussian16
```

### SSH配置
```bash
# Windows PowerShell (交互式登录，弹窗输入密码+2FA)
powershell.exe -Command "ssh -m hmac-sha2-512 twsuday816@ln01.twcc.ai"

# WSL配置别名
Host twcc
    HostName ln01.twcc.ai
    User twsuday816
    IdentityFile ~/.ssh/id_rsa
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 1h
    MACs hmac-sha2-512
    ServerAliveInterval 60

# WSL连接 (密钥认证)
wsl ssh twcc "command"
```

---

## ⚠️ 使用规则与注意事项

### ❌ 绝对禁止

#### 1. 在登录节点运行GPU任务
```bash
后果:
  └─ GPU进程运行超过5分钟 → 自动清除所有进程
  └─ 影响其他用户使用
  └─ 可能导致账户限制

正确做法:
  ✅ sbatch job_script.sh  # 提交到计算节点
  ✅ salloc --gpus=1 --time=01:00:00  # 分配交互式节点
```

#### 2. 在登录节点运行高CPU任务
```bash
后果:
  └─ CPU使用率超过400% → 自动终止进程

正确做法:
  ✅ 所有计算任务通过Slurm提交
```

#### 3. 忽略存储配额
```bash
问题:
  └─ /work 仅100GB，容易超限
  └─ /work 定期清理，数据会丢失
  └─ 大数据集无法存储

解决方案:
  ✅ 训练数据按需下载
  ✅ 重要结果及时备份
  ✅ 使用外部存储或云存储
```

### ✅ 推荐做法

```bash
# 1. 先用gtest调通脚本 (30分钟内，可能免费或优惠)
sbatch -p gtest test_script.sh

# 2. 验证后再用gp1d/gp2d正式训练
sbatch -p gp1d train_script.sh

# 3. 合理估算资源，避免浪费
#SBATCH --gpus-per-node=1  # 需要1个GPU就用1个
#SBATCH --cpus-per-task=4   # 按比例分配CPU
#SBATCH --mem=90G          # 按比例分配内存

# 4. 设置合理的时间
#SBATCH --time=04:00:00  # 实际需要4小时就写4小时

# 5. 重要结果及时备份
# 训练完成后立即下载到本地
```

---

## 💰 费用优化建议

### 💡 莉可丽丝的省钱建议

既然手头预算有限，必须像精打细算：

#### 1. 先用 gtest 调通脚本
```bash
优势:
  ✅ 测试分区通常有扣费优惠（甚至不扣费，取决于TWCC当下政策）
  ✅ 30分钟足够确认Python环境和路径是否正确
  ✅ 快速迭代，不浪费主要配额

流程:
  1. 本地写好脚本
  2. gtest Queue提交 (30分钟)
  3. 确认无错误
  4. gp1d/gp2d正式提交
```

#### 2. 避免申请不必要的资源
```bash
错误示例:
  #SBATCH --gpus-per-node=8  # ❌ 只需要4个GPU却申请8个
  #SBATCH --cpus-per-task=36 # ❌ 浪费CPU资源

正确示例:
  #SBATCH --gpus-per-node=1  # ✅ 实际需要1个就用1个
  #SBATCH --cpus-per-task=4   # ✅ 按比例1:4分配

原因:
  └─ 申请整节点会按整节点计费
  └─ 预算有限时，400元可能10分钟就烧完
```

#### 3. 算好账再"对线"
```bash
推荐流程:
  1. 先跑1个样本/小批次
  2. 统计实际消耗的SU (Service Units)
  3. 估算完整训练需求
  4. 拿数据找教授/导师申请预算

示例话术:
  "老师，我测算了一下，处理完全部样本需要5000 SU，
   现在的500 SU不够塞牙缝的。"
```

### 费用对比: TWNIA2 vs 晶创25

| 项目 | TWNIA2 (V100) | 晶创25 (H100) | 晶创25 (H200) |
|------|--------------|---------------|---------------|
| **GPU性能** | 1x (基准) | ~3x | ~4.5x |
| **显存** | 32GB | 80GB | 141GB |
| **成本** | 低 (约1-3元/SU) | 120元/GPU小时 | 150元/GPU小时 |
| **适用** | 测试验证 | 中大型训练 | 超大模型训练 |

**成本优化策略**:
```
预算 < 500元:
  └─→ TWNIA2 gtest调通 + gp1d小规模训练

预算 500-5000元:
  └─→ TWNIA2完整训练 (时间充足)
  └─→ 晶创25快速完成 (时间紧迫)

预算 > 5000元:
  └─→ 晶创25 H100/H200 (最佳性能)
```

---

## 📚 常用命令速查

### 连接与登录
```bash
# WSL连接 (推荐)
wsl ssh twcc "command"

# PowerShell交互式登录 (弹窗输入密码+2FA)
powershell.exe -Command "ssh -m hmac-sha2-512 twsuday816@ln01.twcc.ai"

# 查看连接状态
wsl ssh twcc "hostname && whoami"
```

### 作业管理
```bash
# 查看队列状态
squeue -u $USER

# 查看节点状态
sinfo -N -o "%.6a %.6A %.10l %.16D %P"

# 提交作业
sbatch job_script.sh

# 取消作业
scancel <job_id>

# 查看作业详情
scontrol show job <job_id>

# 查看作业历史
sacct -u $USER --format=JobID,JobName,State,Elapsed,NNODES
```

### 资源监控
```bash
# GPU状态
nvidia-smi

# 磁盘使用
df -h /work/twsuday816
df -h /home/twsuday816

# 目录大小
du -sh /work/twsuday816/*

# 内存使用
free -h
```

### 数据传输
```bash
# 从本地上传
scp local_file.txt twsuday816@ln01.twcc.ai:/work/twsuday816/

# 下载到本地
scp twsuday816@ln01.twcc.ai:/work/twsuday816/result.txt ./

# WSL方式
wsl scp /mnt/d/local_file.txt twcc:/work/twsuday816/
```

---

## 🚀 快速上手

### 第一次使用流程

#### 1. 登录集群
```bash
# WSL方式 (推荐，密钥认证)
wsl ssh twcc

# PowerShell方式 (交互式登录)
powershell.exe -Command "ssh -m hmac-sha2-512 twsuday816@ln01.twcc.ai"
```

#### 2. 加载环境
```bash
# 查看可用模块
module avail

# 加载CUDA和Python
module load cuda/12.8
module load miniconda3/conda24.5.0_py3.9

# 验证GPU可用
nvidia-smi
```

#### 3. 创建测试脚本
```bash
cd /work/twsuday816

cat > test_gpu.py << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    # 创建测试tensor
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"Matrix multiplication: {z.shape}")
EOF
```

#### 4. 创建作业脚本
```bash
cat > quick_test.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=quick_test
#SBATCH --partition=gtest
#SBATCH --account=ENT114035
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/work/twsuday816/test_%j.out
#SBATCH --error=/work/twsuday816/test_%j.err

module load cuda/12.8
module load miniconda3/conda24.5.0_py3.9

python /work/twsuday816/test_gpu.py
EOF

chmod +x quick_test.sh
```

#### 5. 提交作业
```bash
# 提交测试作业
sbatch quick_test.sh

# 查看状态
squeue -u $USER

# 查看输出
tail -f /work/twsuday816/test_*.out
```

---

## 🔄 与晶创25对比

### 硬件对比
| 项目 | 晶创25 (Nano 5) | TWNIA2 (Taiwania 2) |
|------|-----------------|---------------------|
| **GPU型号** | H100 80GB / H200 141GB | V100 32GB |
| **GPU数量** | 144 (80×H100 + 64×H200) | 2,016 × V100 |
| **峰值性能** | ~30 PFLOPS | 9 PFLOPS |
| **显存/节点** | 640GB-1.1TB (8×GPU) | 256GB (8×GPU) |
| **内存/节点** | 1.9 TB | 768 GB |
| **网络** | HDR IB 200Gbps | EDR IB 100Gbps |
| **存储** | /home 501TB + /work 3.6PB | /home 100GB + /work 100GB |

### 成本对比
| 项目 | 晶创25 (企业/个人) | TWNIA2 (预估) |
|------|-------------------|---------------|
| **H100** | 120元/GPU小时 | - |
| **H200** | 150元/GPU小时 | - |
| **V100** | - | ~1-3元/SU (待确认) |
| **优势** | 高性能、快速完成 | 低成本、GPU多 |
| **适用** | 大模型、高性能计算 | 测试、中小规模训练 |

### 使用建议
```
选择晶创25如果:
  ✅ 模型 > 3B参数
  ✅ 显存需求 > 32GB
  ✅ 需要快速完成训练
  ✅ 预算充足 (> 5000元)

选择TWNIA2如果:
  ✅ 快速测试/调试 (< 30分钟)
  ✅ 预算有限 (< 1000元)
  ✅ 中小规模训练
  ✅ 显存需求 < 32GB

推荐流程:
  1. TWNIA2 gtest调通脚本 (免费/优惠)
  2. TWNIA2 gp1d小规模验证 (低成本)
  3. 晶创25正式训练 (高性能)
```

---

## 📞 获取帮助

### 联系方式
```
官方文档: https://man.twcc.ai/@twccdocs/doc-twnia2-main-zh/
手册: https://man.twcc.ai
客服: 通过TWCC官网联系
```

### 在线资源
```
Queue说明:
  https://man.twcc.ai/@twccdocs/guide-twnia2-queue-zh

计算资源:
  https://man.twcc.ai/@twccdocs/guide-twnia2-compute-capability-zh

存储资源:
  https://man.twcc.ai/@twccdocs/guide-twnia2-storage-capability-zh

Slurm命令:
  https://man.twcc.ai/@twccdocs/guide-twnia2-sbatch-zh
```

---

## 📝 附录

### 环境变量
```bash
# Slurm环境变量
$SLURM_JOB_ID        # 当前作业ID
$SLURM_JOB_NODELIST  # 分配的节点列表
$SLURM_SUBMIT_DIR    # 提交目录
$SLURM_CPUS_PER_TASK # 每任务CPU数
$CUDA_VISIBLE_DEVICES # 可见GPU列表
```

### 推荐的.bashrc配置
```bash
# 添加到 ~/.bashrc
alias sq='squeue -u $USER'
alias gpu='nvidia-smi'
alias work='cd /work/twsuday816'
alias mod='module'
alias ml='module load'

# TWCC快捷方式
alias twcc='wsl ssh twcc'
```

### SSH配置 (WSL)
```bash
# 添加到 ~/.ssh/config
Host twcc
    HostName ln01.twcc.ai
    User twsuday816
    IdentityFile ~/.ssh/id_rsa
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 1h
    MACs hmac-sha2-512
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

---

**文档结束**

本文档基于2026-02-23的实际集群状态编写，配置可能会有更新，请以官方手册为准。
