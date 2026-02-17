# 晶创25 (Nano 5) HPC集群完整介绍

> **文档版本**: v1.0
> **更新日期**: 2026-02-14
> **集群**: 晶创25 (Nano 5)
> **所属**: 国家高速网络与计算中心 (NCHC)
> **位置**: 台湾台中

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
9. [常用命令](#常用命令)
10. [快速上手](#快速上手)
11. [性能优化建议](#性能优化建议)

---

## 🚀 快速概览

### 基本信息
- **集群名称**: 晶创25 (Nano 5)
- **管理机构**: 国家高速网络与计算中心 (NCHC)
- **所属**: 国家实验研究院
- **登录地址**: `nano5.nchc.org.tw` / `twnia3.nchc.org.tw`
- **操作系统**: Red Hat Enterprise Linux 8.10
- **调度系统**: Slurm
- **网络**: 国网高速网络 + InfiniBand

### 规模统计
```
┌─────────────────────────────────────────────┐
│ 节点总数: 18个计算节点                  │
│ GPU总数:   144片 (80×H100 + 64×H200)   │
│ CPU总数:   2,016核 (18×112核)           │
│ 内存总量:   34.2 TB (18×1.9TB)           │
│ 存储总量:   4+ PB                        │
└─────────────────────────────────────────────┘
```

### 分区划分
| 分区 | 节点范围 | GPU类型 | 节点数 | GPU总数 |
|------|----------|---------|---------|---------|
| **normal** | hgpn02-06, 17-21 | H100 80GB | 10 | 80 |
| **normal2** | hgpn39-46 | H200 | 8 | 64 |
| **4nodes** | hgpn02-05 | H100 80GB | 4 | 专用 |
| **dev** | hgpn02-21 | H100/H200 | - | 开发测试 |

---

## 💻 硬件配置

### 计算节点规格

#### H100 节点 (normal分区)
```
节点数量: 10个 (hgpn02-06, 17-21)
每个节点:
  ├─ CPU: Intel Xeon, 112核 (56核×2插槽)
  ├─ 内存: 1.9 TB
  ├─ GPU: 8× NVIDIA H100 80GB HBM3
  ├─ 互联: HDR InfiniBand 200Gbps
  └─ 架构: x86_64

网络配置:
  ├─ 管理: 172.21.100.x (bond0)
  ├─ IB:   ib0, ib1 (10.220.101.x)
  └─ 延迟: 0.03-0.07 ms (节点间)
```

#### H200 节点 (normal2分区)
```
节点数量: 8个 (hgpn39-46)
每个节点:
  ├─ CPU: Intel Xeon, 112核 (56核×2插槽)
  ├─ 内存: 1.9 TB
  ├─ GPU: 8× NVIDIA H200
  ├─ 互联: HDR InfiniBand 200Gbps
  └─ 架构: x86_64
```

### 节点命名规则
```
hg = HPC GPU
pn = 计算节点编号
02-46 = 节点序号

示例: hgpn02 = 第2号HPC GPU节点
```

---

## 🌐 网络架构

### InfiniBand (计算网络)
```
硬件: Mellanox ConnectX-7 (MT4129)
速率: HDR 200 Gbps (部分端口 100 Gbps)
状态: 全部 Active LinkUp

设备:
  ib0 → mlx5_0 (HDR 200Gbps)
  ib1 → mlx5_1 (100Gbps)
  bond1 → 聚合高带宽

延迟测试:
  本地回环: 0.019ms
  节点间:   0.03-0.07 ms
  HiNet DNS: 2.52ms
  Google DNS: 6.48ms
```

### 外网连接
```
公网IP: 140.110.148.x
位置: 台中, 台湾
带宽: ~4 Mbps (学术用途)
延迟: 2-6ms (国内)

限制: 适合SSH/代码传输，不适合大数据下载
```

---

## 💾 存储系统

### 文件系统概览
```
/home (WekaFS):
  容量: 501 TB
  已用: 104 TB (21%)
  位置: /home/twsuday816
  用途: 个人文件、配置、SSH密钥
  特点: 长期存储、系统备份

/work (高性能并行FS):
  容量: 3.6 PB
  已用: 389 TB (11%)
  位置: /work/twsuday816
  用途: 训练脚本、作业输出、工作数据
  特点: 高IOPS、跨节点共享
```

### 目录使用建议
```
推荐结构:

/work/twsuday816/
├── scripts/      # 训练脚本
├── data/        # 输入数据集
├── models/      # 保存的模型
├── results/     # 输出结果
└── logs/        # 日志文件

/home/twsuday816/
├── .ssh/        # SSH密钥
├── .local/      # Python包安装位置
└── .config/     # 配置文件
```

### 配额情况
```
用户目录 (home): 8.3 GB 已用 / 501 TB 总量
工作目录 (work): 72 KB 已用 / 3.6 PB 总量

建议:
  ✅ 训练数据放 /work (高性能)
  ✅ 脚本放 /work
  ✅ 大模型放 /work
  ❌ 避免在 ~ 放大量数据
```

---

## 🔧 软件环境

### 模块系统 (Lmod)
```
可用模块类别:

编译器:
  gcc: 8.5.0, 10.5.0, 11.5.0, 12.5.0
  Intel Compiler: 2024.2.1 (ifort, icx, icpx)

CUDA:
  cuda: 11.6, 12.2, 12.4(默认), 12.6

MPI:
  openmpi: 4.1.6, 5.0.5
  Intel MPI: 2021.13

数学库:
  Intel MKL: 2024.2
  Intel DNNL: 3.5.0
  Intel CCL: 2021.13

容器:
  Singularity: 3.7.1

Python:
  Miniconda3: 24.11.1 (Python 3.12.9)

性能工具:
  Intel VTune: 2024.2
  Intel Advisor: 2024.2
  Intel Debugger: 2024.2
```

### 用户安装的Python包
```
位置: ~/.local/lib/python3.12/site-packages/

深度学习:
  torch 2.6.0+cu124
  torchvision 0.21.0+cu124
  torchaudio 2.6.0+cu124

Hugging Face:
  transformers 5.1.0
  datasets 4.5.0
  tokenizers 0.22.2
  accelerate 1.12.0

其他:
  numpy, pandas, pyarrow, safetensors
  tqdm, rich, pydantic
```

---

## ⚙️ Slurm作业调度

### 基本概念
```
Partition (分区): 节点分组，如 normal, normal2
QoS (服务质量): 作业优先级和服务类别
Account (账户): 资源分配单位
Job (作业): 计算任务
JobID: 作业唯一标识符
TRES (跟踪资源): CPU, 内存, GPU, billing
```

### 优先级权重
```
PriorityWeightFairShare: 1,000,000 (最高权重)
PriorityWeightQOS:      100,000
PriorityWeightAge:      10,000
PriorityWeightPartition: 10,000
PriorityWeightJobSize:   10,000
PriorityDecayHalfLife:  7天

→ 公平分享优先，鼓励合理使用
→ 新作业有优势(7天衰减)
```

### 资源限制
```
MaxJobCount: 50,000
MaxNodes: 2个/作业 (normal分区)
MaxTime:   48小时
MaxTasksPerNode: 512

节点限制:
  H100: 8 GPU/节点
  H200: 8 GPU/节点
```

### ⚠️ 弹性节点配置行为

**重要说明**: normal分区虽然支持 `--nodes=1-2` 语法，但由于集群配置，**实际不会按预期工作**。

#### 集群配置影响
```bash
# 分区配置
SelectTypeParameters = CR_PACK_NODES  # 打包节点到最少数量
SchedulerType      = sched/backfill   # 回填调度器
MaxNodes           = 2                 # normal分区硬限制
```

#### 实际行为
```bash
# ❌ 弹性配置（不推荐）
#SBATCH --nodes=1-2
# 问题：会等待2个节点都可用才启动，不会先用1个节点

# ✅ 固定1节点（推荐：测试/调试）
#SBATCH --nodes=1
# 优势：可立即运行，无需等待

# ✅ 固定2节点（正式训练）
#SBATCH --nodes=2
# 说明：需要等待2个节点同时可用
```

#### 验证节点配置
```bash
# 查看作业实际节点需求
scontrol show job <JOB_ID> -o | grep NumNodes
# 输出示例: NumNodes=2-2  (表示固定2节点，不是1-2)

# 查看分区限制
scontrol show partition normal | grep -E "MaxNodes|MinNodes"
```

#### 推荐做法
1. **快速测试**: 使用固定1节点，立即验证代码
2. **调试阶段**: 使用固定1节点，节省等待时间
3. **正式训练**: 使用固定2节点，获得最佳性能
4. **避免使用**: `--nodes=1-2`（在此集群上不会带来弹性优势）

### 常用Slurm命令
```bash
# 查看队列
squeue -u $USER

# 查看节点状态
sinfo -N -l

# 提交作业
sbatch job_script.sh

# 交互式会话
srun --pty bash

# 取消作业
scancel <job_id>

# 查看作业历史
sacct -u $USER --format=JobID,JobName,State,Elapsed

# 查看账户信息
sacctmgr show associations where User=$USER
```

---

## 👤 账户与权限

### 账户类型
```
主账户: twsuday816 (UID: 204634)

关联账户:
  ent114035 (企业账户, GID: 102622)
  tri900137 (国研院账户, GID: 102640)

默认组: TRI900137
Shell: /bin/bash
Home: /home/twsuday816
```

### 账户状态
```
总作业数: 72个 (2026年2月)
成功率: 100% (所有COMPLETED)
计费单位: billing=1+ (按GPU小时计费)
QOS: normal
```

### 邮件通知设置
```bash
# 在脚本中添加:
#SBATCH --mail-type=END,FAIL    # 作业结束或失败时邮件
#SBATCH --mail-user=your@email.com  # 你的邮箱地址
```

---

## ⚠️ 使用规则与注意事项

### ❌ 绝对禁止

#### 1. 在登录节点运行计算任务
```
后果:
  - 系统负载过高，影响其他用户
  - 会被管理员直接kill进程
  - 严重违规可能暂停账户
  - 性能极差（无独占GPU）

检测方法:
  - CPU负载监控
  - 进程监控
  - GPU使用监控

正确做法:
  sbatch job_script.sh  # 提交到计算节点
  srun --pty bash      # 交互式计算节点会话
```

#### 2. 频繁查看队列
```
错误做法:
watch -n 1 squeue -u $USER  # ❌ 每秒刷新，增加负载

正确做法:
squeue -u $USER           # ✅ 手动查看
squeue -u $USER -i 10    # ✅ 10秒刷新
wait 30 && squeue -u $USER  # ✅ 30秒间隔
```

#### 3. 使用系统包管理器
```
错误命令:
sudo apt install xxx    # ❌ RHEL用yum
sudo yum install xxx    # ❌ 无权限

正确做法:
module load miniconda3/24.11.1
conda install pytorch       # ✅ 用conda安装
pip install xxx --user    # ✅ 安装到用户目录
```

### ✅ 推荐做法

```bash
# 1. 所有计算通过Slurm
sbatch run_training.sh

# 2. 数据放/work目录
/work/twsuday816/data/

# 3. 作业输出定向到/work
#SBATCH --output=/work/twsuday816/result_%j.out
#SBATCH --error=/work/twsuday816/result_%j.err

# 4. 合理估算资源
#SBATCH --time=24:00:00     # 实际需要的时间
#SBATCH --gpus-per-node=1   # 实际需要的GPU数

# 5. 使用checkpoint
# 长任务保存中间结果
```

---

## 📚 常用命令速查

### 环境管理
```bash
# 查看可用模块
module avail

# 加载模块
module load cuda/12.6
module load miniconda3/24.11.1

# 查看已加载模块
module list

# 卸载模块
module purge
```

### 作业管理
```bash
# 提交批处理作业
sbatch [options] script.sh

# 交互式分配资源
srun --partition=normal --account=ACCOUNT \
      --nodes=1 --gpus-per-node=1 \
      --time=01:00:00 --pty bash

# 查看队列
squeue -u $USER

# 取消作业
scancel JOBID

# 查看作业详情
scontrol show job JOBID
```

### 资源监控
```bash
# GPU状态
nvidia-smi

# 节点状态
sinfo -N -l

# 磁盘使用
df -h /work/twsuday816

# 目录大小
du -sh /work/twsuday816

# 内存使用
free -h
```

### 数据传输
```bash
# 从本地上传
scp local_file.txt twsuday816@nano5.nchc.org.tw:/work/twsuday816/

# 下载到本地
scp twsuday816@nano5.nchc.org.tw:/work/twsuday816/result.txt ./

# 同步目录
rsync -avz --progress local_dir/ \
  twsuday816@nano5.nchc.org.tw:/work/twsuday816/
```

### 网络测试
```bash
# 节点间延迟
ping hgpn02

# 查看节点信息
sinfo -n -l -o "%all"

# InfiniBand状态
ibstat
```

---

## 🚀 快速上手

### 第一次使用流程

#### 1. 登录集群

**Windows + WSL (推荐):**
```bash
# 在WSL终端中执行
wsl ssh twsuday816@nano5.nchc.org.tw
# 或
wsl ssh twsuday816@twnia3.nchc.org.tw
```

**Linux/macOS 直接登录:**
```bash
ssh twsuday816@nano5.nchc.org.tw
# 或
ssh twsuday816@twnia3.nchc.org.tw
```

**配置SSH密钥 (避免每次输入密码):**
```bash
# 在本地生成密钥
ssh-keygen -t rsa -b 4096

# 复制公钥到服务器
ssh-copy-id -i ~/.ssh/id_rsa.pub twsuday816@nano5.nchc.org.tw

# 后续登录无需密码
wsl ssh twsuday816@nano5.nchc.org.tw
```

#### 2. 加载环境
```bash
# 查看可用环境
module avail

# 加载CUDA和Python
module load cuda/12.6
module load miniconda3/24.11.1

# 验证GPU可用
nvidia-smi
```

#### 3. 准备训练脚本
```bash
cd /work/twsuday816

# 创建示例脚本
cat > train_simple.py << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

#### 4. 创建作业脚本
```bash
cat > simple_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test_train
#SBATCH --partition=normal
#SBATCH --account=ENT114035
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/work/twsuday816/result_%j.out
#SBATCH --error=/work/twsuday816/result_%j.err

module load cuda/12.6
module load miniconda3/24.11.1

python /work/twsuday816/train_simple.py
EOF

chmod +x simple_job.sh
```

#### 5. 提交作业
```bash
sbatch simple_job.sh

# 查看状态
squeue -u $USER

# 查看输出
tail -f /work/twsuday816/result_*.out
```

---

## ⚡ 性能优化建议

### GPU选择策略

```python
# H100 vs H200 选择

选择H100如果:
  ✅ 模型显存需求 < 80GB
  ✅ batch_size适中
  ✅ 需要快速迭代
  ✅ 队列通常较短

选择H200如果:
  ✅ 超大模型(>70B参数)
  ✅ 极大batch_size
  ✅ 长上下文推理
  ✅ 推理密集型任务

通用建议:
  → 先用H100测试 (80%场景够用)
  → 显存不足再考虑H200
```

### 数据加载优化
```python
# ✅ 好的做法
# 数据放/work (高性能存储)
dataset = load_dataset('/work/twsuday816/data/train')

# 使用多worker加速
dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)

# ❌ 避免
dataset = load_dataset('/home/twsuday816/data/train')  # 性能差
dataloader = DataLoader(dataset, num_workers=0)      # 单线程
```

### 训练优化
```bash
# 1. 使用合适的GPU数量
#SBATCH --gpus-per-node=1  # 单GPU任务
#SBATCH --gpus-per-node=4  # 多GPU并行训练

# 2. 批量大小根据显存调整
# H100 80GB: batch_size=128-512 较安全
# 考虑模型占用和优化器状态

# 3. 混合精度训练
# 使用FP16/BF16加速训练
model = model.half()  # PyTorch
with torch.cuda.amp.autocast():  # 自动混合精度
```

### 作业提交优化
```bash
# 1. 合理估算时间
#SBATCH --time=04:00:00  # 不要估算过多
# 估算少了会被kill
# 估算多了会排队更久

# 2. 使用邮件通知
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com

# 3. 输出到合适位置
#SBATCH --output=/work/twsuday816/results/%j.out
#SBATCH --error=/work/twsuday816/results/%j.err
```

---

## 📊 性能基准

### 训练性能参考
```
实测案例 (H100, PyTorch 2.6.0):
模型: SimpleGPT (13.26M参数)
数据: 随机数据
批次: 32
序列: 64

结果:
  训练时间: 1.06秒 (200批次)
  吞吐量: 6,029 样本/秒
  Token吞吐: 385,844 tokens/秒
  每批次: 5.31 ms

估算:
  ImageNet (1.28M images): ~3.5分钟/epoch
  GPT-3 small (训练): ~数小时完成
```

### 网络性能
```
节点间通信:
  延迟: 0.03-0.07 ms (优秀)
  带宽: 200 Gbps (HDR InfiniBand)
  抖动: 0.01-0.02 ms (极低)

外网连接:
  下载: ~1.7 Mbps (受限)
  上传: ~0.9 Mbps
  延迟: 2-6 ms (台湾ISP)

适用场景:
  ✅ Git代码拉取
  ✅ pip/conda安装包
  ❌ 大数据集下载 (建议其他方式)

### 实测传输性能
本地→Nano5 (数据传输端口2222):
  上传速度: 12MB/s (96 Mbps)
  预估上传时间:
    - 1GB: ~1.5分钟
    - 10GB: ~14分钟
    - 20GB (1B模型训练数据): ~30分钟
    - 100GB: ~2.3小时

  推荐传输方式:
    - 小文件 (<1GB): scp / sftp
    - 大文件 (>1GB): rsync --progress (支持断点续传)
    - 批量传输: tar打包后传输 (减少IO次数)

外网受限 (学术用途):
  下载: ~1.7 Mbps
  上传: ~0.9 Mbps
  建议大数据集先下载到本地，再从本地上传到Nano5
```

---

## 💰 计费标准

### GPU计费 (2025年3月20日开始正式收费)

> 计算单位：**GPU小时** = 执行小时数 × GPU数量
>
> 最低申请金额：500元

#### 晶创25 H100 (normal分区)

| 计划类别 | 每GPU小时 |
|----------|-----------|
| 国科会计划 | 25元 |
| 学术计划 | 50元 |
| 政府与法人计划 | 50元 |
| **企业与个人计划** | **120元** |

#### 晶创25 H200 (normal2分区)

| 计划类别 | 每GPU小时 |
|----------|-----------|
| 国科会计划 | 30元 |
| 学术计划 | 60元 |
| 政府与法人计划 | 60元 |
| **企业与个人计划** | **150元** |

> **注意**: 以上费用为NCHC国家计费系统标准（2025年3月），具体可能因政策调整而变动

### 用户账户类型说明

当前用户 (twsuday816) 关联账户：
- **ent114035** (企业账户) → 适用企业与个人计划费率
- **tri900137** (国研院账户) → 可能适用学术计划费率

建议：确认您的账户类型以准确估算成本

### GPU性能对比

| GPU类型 | FP64性能 | FP16/BF16 | 显存 | 相对性能 |
|---------|----------|-----------|------|----------|
| H100 | ~30 TFLOPS | ~60 TFLOPS | 80GB HBM3 | 1.0x |
| H200 | ~70 TFLOPS | ~140 TFLOPS | 141GB HBM3e | ~2.3x |

**关键结论**: H200性能略快约10%，但单价高25%，总体成本接近或略高于H100。选择H200主要优势是：显存更大(141GB vs 80GB)、适合超大batch或长序列训练

### 1B模型训练成本估算

#### 计算说明

**估算方法**：基于FLOPs计算（6×参数量×tokens数）

```
Total FLOPs ≈ 6 × 1B参数 × 20B tokens = 1.2×10²⁰ FLOPs

8×H100有效算力 ≈ 1.2-2.0 PFLOPS (BF16，考虑通信/IO开销)
→ 训练时间 ≈ 16.7-27.8 小时

8×H200有效算力 ≈ 1.3-2.2 PFLOPS (BF16，考虑通信/IO开销)
→ 训练时间 ≈ 15.0-26.0 小时
```

> **注意**：实际时间取决于训练吞吐量（tokens/s），建议在集群上跑10分钟短测获取准确数值

#### 场景: 从头预训练，Chinchilla推荐规模（20B tokens）

**数据规模**:
- C4 English: ~236GB (allenai/c4)
- C4 Chinese: ~3.5GB (shjwudp/chinese-c4)
- 总计: ~240GB

**训练配置**:
- 模型: 1B参数
- 序列长度: 2048
- Global batch size: 512 (64×8 GPU)
- 训练tokens: 20B (Chinchilla推荐)

##### 企业与个人计划

| 方案 | 训练时间 | GPU小时 | 总成本 |
|------|----------|---------|--------|
| **H100** 8×GPU | 16.7-27.8小时 | 134-222 | **16,080 - 26,640元** |
| **H200** 8×GPU | 15.0-26.0小时 | 120-208 | **18,000 - 31,200元** |

**对比**: H200时间接近但略快，成本因单价高10-25%

##### 国科会计划

| 方案 | 训练时间 | GPU小时 | 总成本 |
|------|----------|---------|--------|
| **H100** 8×GPU | 16.7-27.8小时 | 134-222 | **3,350 - 5,550元** |
| **H200** 8×GPU | 15.0-26.0小时 | 120-208 | **3,600 - 6,240元** |

##### 学术/政府计划

| 方案 | 训练时间 | GPU小时 | 总成本 |
|------|----------|---------|--------|
| **H100** 8×GPU | 16.7-27.8小时 | 134-222 | **6,700 - 11,100元** |
| **H200** 8×GPU | 15.0-26.0小时 | 120-208 | **7,200 - 12,480元** |

#### 不同训练规模成本对比 (企业与个人计划)

| 训练规模 | 预估tokens | H100时间 | H100成本 | H200时间 | H200成本 | 对比 |
|----------|------------|----------|----------|----------|----------|------|
| 小型 | 5B | 4.2-7.0h | 4,020-6,720元 | 3.8-6.5h | 4,560-7,800元 | H200略快稍贵 |
| 标准 | 20B (Chinchilla) | 16.7-27.8h | 16,080-26,640元 | 15.0-26.0h | 18,000-31,200元 | 时间接近，成本接近 |
| 大型 | 50B | 41.8-69.5h | 40,160-66,480元 | 37.5-65.0h | 45,000-78,000元 | H200快10% |
| 超大 | 100B | 83.5-139h | 80,320-132,960元 | 75.0-130.0h | 90,000-156,000元 | H200快10% |

> **重要提醒**:
> - MaxTime限制：单次作业最长48小时
> - 20B tokens训练：H100和H200都可在48小时内完成，无需分段
> - 大型训练（50B+）：需要checkpoint续跑或使用H200+VB优化
> - 数据上传：训练20B tokens约需20GB数据，上传时间约30分钟
> - H200优势：训练时间略快（约10%），但单价高25%，成本接近或略高
> - VirtualBlackwell可节省10-20%训练时间，降低总成本
> - 确认您的账户类型以使用正确费率

### 费用优化建议

1. **H100 vs H200选择**：
   - 1B模型、20B tokens：两者成本接近，H100略便宜
   - H200适用场景：显存需求>80GB、超大batch size、长序列(>4096)
   - 一般情况优先使用H100，除非特殊需求

2. **使用VirtualBlackwell**: 100% cache hit率可减少10-20%训练时间，降低总成本

3. **合理设置--time**: 避免估算过多导致额外计费

4. **检查点优化**: 定期保存checkpoint，避免训练失败重跑

5. **数据预处理**: 提前完成数据清洗和tokenization

---

## 🔍 故障排查

### 常见问题

#### 作业一直Pending
```bash
# 原因排查
scontrol show job JOBID  # 查看原因

# 常见原因:
  Resources: 节点资源不足
  Priority: 优先级不够高
  QOSMaxGRES: 超过QOS限制
  AssocGrpGRES: 超过账户限制

解决方案:
  - 调整 --time 减少时间
  - 切换分区 (normal ↔ normal2)
  - 检查账户配额
```

#### 作业被Kill
```bash
# 查看作业状态
sacct -j JOBID -o JobID,State,ExitCode

# ExitCode含义:
  0:0   → 正常完成
  1:0   → 应用错误
  9:0   → 内存超限
  125:0 → 信号终止

常见原因:
  - 内存不足 (OOM)
  - 超时 (TIME_LIMIT)
  - 节点故障
```

#### CUDA错误
```bash
# 检查GPU
nvidia-smi

# 常见问题:
  CUDA_OUT_OF_MEMORY: 减小batch_size
  libcudart错误: 检查CUDA版本兼容性
  Device not available: 检查GPU分配
```

---

## 📞 获取帮助

### 联系方式
```
管理员: NCHC团队
手册: https://man.twcc.ai/@AI-Pilot/manual
Email: 通过NCHC官网联系
```

### 在线资源
```
官方手册:
  https://man.twcc.ai/@AI-Pilot/manual

教育视频:
  Part 1: https://www.youtube.com/watch?v=2XOoBXvyRNA
  Part 2: https://www.youtube.com/watch?v=RNE3fl3HHXA
  Part 3: https://www.youtube.com/watch?v=WkBxv6eKrMo

Slurm文档:
  https://slurm.schedmd.com/documentation.html
```

### 常用命令快速参考
```bash
# 查看所有环境模块
module spider

# 查看Python包
pip list | grep -i torch

# 查看作业
myjobs  # 自定义别名
alias myjobs='squeue -u $USER -o "%.18i %.9P %.50j %.2t %.10M %.6D %R"'

# 快速查看GPU
alias gpu='nvidia-smi'
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

# CUDA环境变量
$CUDA_VISIBLE_DEVICES # 可见GPU列表(如0,1,2,3)
```

### 节点列表速查
```
H100节点 (normal):
  hgpn02, hgpn03, hgpn04, hgpn05, hgpn06
  hgpn17, hgpn18, hgpn19, hgpn20, hgpn21

H200节点 (normal2):
  hgpn39, hgpn40, hgpn41, hgpn42
  hgpn43, hgpn44, hgpn45, hgpn46
```

### 推荐的.bashrc配置
```bash
# 添加到 ~/.bashrc
alias sq='squeue -u $USER'
alias gpu='nvidia-smi'
alias work='cd /work/twsuday816'
alias mod='module'
alias ml='module load'

# 添加环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$HOME/.local/bin:$PATH
```

---

**文档结束**

本文档基于2026-02-14的实际集群状态编写，配置可能会有更新，请以官方手册为准。
