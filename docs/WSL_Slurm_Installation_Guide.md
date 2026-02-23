# 在WSL上安装Slurm（测试/学习用）

> 警告：Slurm是集群管理器，WSL单机环境不需要
> 此安装仅用于学习和测试目的

## 安装步骤

### 1. 安装依赖

```bash
sudo apt update
sudo apt install -y build-essential \
    libmunge-dev \
    libmunge2 \
    libssl-dev \
    libpam0-dev \
    libjson-c-dev \
    libhttp-parser-dev \
    python3-pip
```

### 2. 安装Munge（认证服务）

```bash
# 安装munge
sudo apt install -y munge

# 创建munge用户和组
sudo useradd -r -s /bin/false munge
sudo groupadd munge
sudo usermod -a -G munge munge

# 设置权限
sudo chown -R munge:munge /etc/munge /var/lib/munge /var/log/munge /run/munge

# 生成munge key
sudo /usr/sbin/create-munge-key -f

# 设置权限
sudo chmod 400 /etc/munge/munge.key
sudo chmod 700 /etc/munge
```

### 3. 下载并编译Slurm

```bash
# 下载slurm源码（以slurm 23.02为例）
cd ~
wget https://download.schedmd.com/slurm/slurm-23.02.4.tar.bz2

# 解压
tar xjf slurm-23.02.4.tar.bz2
cd slurm-23.02.4

# 配置（WSL单机配置）
./configure --prefix=/usr/local \
    --sysconfdir=/etc/slurm \
    --with-munge \
    --disable-pam \
    --with-json=config \
    --enable-slurmrestd \
    --with-http-parser=none

# 编译（需要一些时间）
make -j$(nproc)

# 安装
sudo make install
```

### 4. 配置Slurm（单机测试配置）

```bash
# 创建配置目录
sudo mkdir -p /etc/slurm

# 创建最小化配置
sudo tee /etc/slurm/slurm.conf << 'EOF'
# slurm.conf - 单机测试配置

ClusterName=WSL
SlurmctldHost=localhost

# 计算节点定义
NodeName=WSL CPUs=4 RealMemory=8000 State=UNKNOWN

# 分区定义
PartitionName=debug Nodes=WSL Default=YES MaxTime=30

# 日志
SlurmctldDebug=3
SlurmctldLogFile=/var/log/slurm/slurmctld.log

# 限制（测试用）
DefMemPerCPU=8000
MaxMemPerCPU=8000
EOF

# 创建日志目录
sudo mkdir -p /var/log/slurm
sudo chown -R $USER:$USER /var/log/slurm
```

### 5. 启动Slurm服务

```bash
# 启动munge
sudo service munge start

# 启动slurmctld（控制器）
sudo /usr/local/sbin/slurmctld

# 启动slurmd（计算节点守护进程）
sudo /usr/local/sbin/slurmd

# 检查状态
sinfo
squeue
```

### 6. 测试Slurm

```bash
# 测试提交作业
cat > test_job.sh << 'EOF'
#!/bin/bash
echo "Job started on $(hostname)"
echo "Running on node: $SLURM_NODELIST"
sleep 5
echo "Job completed"
EOF

chmod +x test_job.sh
sbatch --wrap="python3 --version" test_job.sh

# 查看作业
squeue
```

## 清理（如果需要）

```bash
# 停止服务
sudo pkill slurmd
sudo pkill slurmctld
sudo service munge stop

# 卸载（如果从源码安装）
cd ~/slurm-23.02.4
sudo make uninstall

# 清理munge
sudo apt remove --purge munge libmunge2
sudo userdel munge
sudo groupdel munge
```

## WSL2 GPU配置（重要更新）

### ⚠️ WSL2 GPU设备的特殊性

**核心问题**：WSL2 **不使用**传统的 `/dev/nvidia*` 设备文件！

WSL2使用微软的GPU半虚拟化技术：
- 设备节点：`/dev/dxg`（而不是 `/dev/nvidia0`）
- GPU访问：通过DirectX转换层
- Slurm的GRES插件**不支持**这种设备类型

### 🔧 WSL2 Slurm GPU配置方案

#### 方案1: 绕过Slurm GPU管理（推荐）

让Slurm管理CPU，GPU由应用程序通过CUDA直接访问：

**`/etc/slurm/slurm.conf`**：
```bash
ClusterName=WSL
SlurmctldHost=localhost
NodeName=LAPTOP-V07LDHP3 CPUs=8 RealMemory=10000 State=UNKNOWN
PartitionName=debug Nodes=LAPTOP-V07LDHP3 Default=YES MaxTime=30

# 禁用cgroup（WSL2不支持）
TaskPlugin=task/none

# 使用cons_tres选择器
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# 邮件程序修复
MailProg=/bin/true
```

**`/etc/slurm/gres.conf`**：
```bash
# 核心技巧：File 指向 /dev/null
# Slurm检查设备存在时，/dev/null永远"存在"
# AutoDetect必须移除，否则Slurm会发现实际GPU数量不匹配
Name=gpu Count=1 File=/dev/null
```

**任务脚本中设置GPU**：
```bash
#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=test_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# 关键：在任务脚本中设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 验证GPU可用
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 你的训练代码
python3 train.py --batch-size 4 --gpu 0
```

#### 方案2: 添加GRES支持（实验性，不推荐）

⚠️ **此方案会导致节点INVAL状态**，仅作记录：

```bash
# slurm.conf
GresTypes=gpu
NodeName=LAPTOP-V07LDHP3 ... Gres=gpu:1

# gres.conf
Name=gpu Count=1 File=/dev/dxg  # ❌ Slurmd无法识别
# 或
Name=gpu Count=1                  # ❌ 数量验证失败 (0 < 1)
```

**错误信息**：
```
error: gres/gpu count reported lower than configured (0 < 1)
State changed to INVAL
```

**失败原因**：
1. Slurmd的GPU检测基于NVML库
2. WSL2的nvidia-smi在`/usr/lib/wsl/lib/`，root用户PATH中找不到
3. `/dev/dxg`不是Slurm期望的GPU设备文件类型

### 📋 配置检查清单

- [ ] 禁用cgroup：`TaskPlugin=task/none`
- [ ] 设置SelectType：`SelectType=select/cons_tres`
- [ ] gres.conf使用`File=/dev/null`技巧
- [ ] 任务脚本中设置`CUDA_VISIBLE_DEVICES`
- [ ] 测试：`nvidia-smi`和`python -c "import torch; ..."`都能识别GPU

### 🚀 启动命令

```bash
# 清理ghost job
sudo pkill slurmd; sudo pkill slurmctld
sudo rm -rf /var/spool/slurmctld/*

# 启动服务
sudo /usr/local/sbin/slurmctld
sudo /usr/local/sbin/slurmd -N LAPTOP-V07LDHP3

# 验证
sinfo  # 应该看到节点状态为idle
```

### 💡 最佳实践

1. **本地开发 → 集群训练流程**：
   - WSL Slurm：调试脚本、验证逻辑（免费）
   - Nano5集群：正式训练、完整测试（计费）

2. **GPU使用建议**：
   - 在脚本开头验证CUDA：`torch.cuda.is_available()`
   - 监控GPU使用：另开窗口运行`watch -n 1 nvidia-smi`
   - 准备好`wsl --shutdown`作为紧急停止

3. **故障排除**：
   - 进程D状态（不可中断）：`wsl --shutdown`
   - 节点INVAL状态：检查GRES配置，移除GPU相关配置
   - GPU不可用：检查`CUDA_VISIBLE_DEVICES`和驱动版本

## 注意事项

⚠️ **WSL上运行Slurm的限制**：
- 没有真正的多节点
- 没有cgroup支持
- Slurm无法直接管理GPU资源
- 仅用于学习和测试

✅ **推荐做法**：
- 使用WSL SSH连接真实集群
- 在真实集群上运行生产任务
- WSL Slurm用于脚本验证和逻辑调试
