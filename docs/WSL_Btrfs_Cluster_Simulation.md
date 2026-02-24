# Btrfs模拟集群训练环境指南

> 创建日期: 2026-02-23
> 背景: 使用btrfs文件系统在本地WSL环境中模拟分布式训练集群的特性

## 核心思想

btrfs文件系统具有许多与分布式集群环境相似的特性，可以在本地WSL环境中模拟集群训练场景：

```
集群环境特性                    btrfs对应特性
├── Checkpoint/回滚           → 快照 (Snapshot)
├── 节点间数据同步            → 发送/接收 (Send/Receive)
├── 实验环境隔离              → 子卷 (Subvolume)
├── 存储空间优化              → CoW + 压缩 (Copy-on-Write + Compression)
└── 增量备份                  → 增量快照发送
```

## Btrfs创建与挂载

### 创建btrfs镜像

```bash
# 1. 创建镜像文件（根据需要调整大小）
dd if=/dev/zero of=~/btrfs_image/disk.img bs=1G count=10

# 2. 格式化为btrfs
mkfs.btrfs ~/btrfs_image/disk.img

# 3. 挂载
mkdir -p ~/mnt/btrfs
sudo mount -o loop ~/btrfs_image/disk.img ~/mnt/btrfs

# 4. 查看空间使用
df -h ~/mnt/btrfs
btrfs filesystem df ~/mnt/btrfs
```

### 启用压缩（推荐）

```bash
# 重新挂载启用压缩
sudo umount ~/mnt/btrfs
sudo mount -o compress=zstd ~/mnt/btrfs

# 或者在线启用（需要内核5.19+）
sudo btrfs property set ~/mnt/btrfs compression zstd
```

## 核心特性与应用

### 1. 快照（Snapshot）- 训练Checkpoint回滚

#### 基础操作

```bash
# 创建快照
sudo btrfs subvolume snapshot ~/mnt/btrfs ~/mnt/btrfs/snap_before_train

# 列出快照
sudo btrfs subvolume list ~/mnt/btrfs

# 删除快照
sudo btrfs subvolume delete ~/mnt/btrfs/snap_before_train
```

#### 实际应用场景

```bash
# 场景1: 训练前创建checkpoint
cd ~/mnt/btrfs/APT-Transformer
sudo btrfs subvolume snapshot ~/mnt/btrfs ~/snapshots/baseline_$(date +%Y%m%d_%H%M%S)

# 场景2: 训练失败，瞬间回滚
sudo btrfs subvolume delete ~/mnt/btrfs
sudo btrfs subvolume snapshot ~/snapshots/baseline_20250223_170000 ~/mnt/btrfs

# 场景3: 保留多个实验版本
sudo btrfs subvolume snapshot ~/mnt/btrfs ~/snappoints/exp_v1_lr0.001
# 实验 v2
sudo btrfs subvolume snapshot ~/mnt/btrfs ~/snappoints/exp_v2_lr0.0001
# 对比两个版本
diff -r ~/snappoints/exp_v1_lr0.001/APT-Transformer/checkpoints \
        ~/snappoints/exp_v2_lr0.0001/APT-Transformer/checkpoints
```

#### 只读快照（防止误修改）

```bash
# 创建只读快照
sudo btrfs subvolume snapshot -r ~/mnt/btrfs ~/snappoints/immutable_baseline

# 尝试修改会失败
touch ~/snappoints/immutable_baseline/test.txt  # Permission denied
```

### 2. 发送/接收（Send/Receive）- 节点间数据同步

#### 基础操作

```bash
# 完整发送
sudo btrfs send ~/mnt/btrfs/snap_new | \
  ssh remote-node "sudo btrfs receive /backup/"

# 增量发送（需要父快照）
sudo btrfs send -p ~/mnt/btrfs/snap_base ~/mnt/btrfs/snap_new | \
  ssh remote-node "sudo btrfs receive /backup/"
```

#### 实际应用场景

```bash
# 场景1: 同步checkpoint到集群存储
# 首次完整同步
sudo btrfs subvolume snapshot -r ~/mnt/btrfs ~/snapshots/exp_001_base
sudo btrfs send ~/snapshots/exp_001_base | \
  ssh cluster-storage "sudo btrfs receive /data/experiments/"

# 增量同步新checkpoint
sudo btrfs subvolume snapshot -r ~/mnt/btrfs ~/snapshots/exp_002_new
sudo btrfs send -p ~/snapshots/exp_001_base ~/snapshots/exp_002_new | \
  ssh cluster-storage "sudo btrfs receive /data/experiments/"

# 场景2: 本地增量备份
sudo btrfs send -p ~/snapshots/prev ~/snapshots/current | \
  sudo btrfs receive /backup/experiments/

# 场景3: 压缩发送流
sudo btrfs send ~/snappoints/exp_001 | gzip | \
  ssh backup "gunzip | sudo btrfs receive /backup/"
```

### 3. 子卷（Subvolume）- 实验环境隔离

#### 基础操作

```bash
# 创建子卷
sudo btrfs subvolume create ~/mnt/btrfs/exp_v1
sudo btrfs subvolume create ~/mnt/btrfs/exp_v2

# 挂载子卷
sudo mkdir ~/exp1
sudo mount -o subvol=exp_v1 /dev/loop0 ~/exp1

# 查看子卷
sudo btrfs subvolume list ~/mnt/btrfs
```

#### 实际应用场景

```bash
# 场景1: 并行实验不同配置
# 实验v1: LR=0.001
sudo btrfs subvolume create ~/mnt/btrfs/exp_lr0.001
cp -r ~/APT-Transformer ~/mnt/btrfs/exp_lr0.001/
# 修改config: lr=0.001
vim ~/mnt/btrfs/exp_lr0.001/APT-Transformer/configs/train.yaml

# 实验v2: LR=0.0001
sudo btrfs subvolume create ~/mnt/btrfs/exp_lr0.0001
cp -r ~/APT-Transformer ~/mnt/btrfs/exp_lr0.0001/
# 修改config: lr=0.0001
vim ~/mnt/btrfs/exp_lr0.0001/APT-Transformer/configs/train.yaml

# 场景2: 快速切换实验环境
# 创建软链接指向不同子卷
ln -s ~/mnt/btrfs/exp_lr0.001/APT-Transformer ~/active_exp
# 实验 v1
cd ~/active_exp && python train.py
# 切换到实验 v2
rm ~/active_exp
ln -s ~/mnt/btrfs/exp_lr0.0001/APT-Transformer ~/active_exp
cd ~/active_exp && python train.py

# 场景3: 清理失败的实验
sudo btrfs subvolume delete ~/mnt/btrfs/exp_failed
```

### 4. Copy-on-Write + 压缩 - 空间优化

#### 检查压缩状态

```bash
# 查看压缩统计
sudo btrfs filesystem df -h ~/mnt/btrfs
# 输出示例:
# Data, single: total=1.00GiB, used=800.00MiB
# Data, compressed: total=300.00MiB, used=250.00MiB  (压缩率~70%)

# 查看文件压缩情况
sudo btrfs filesystem du -h ~/mnt/btrfs/APT-Transformer
```

#### 实际应用场景

```bash
# 场景1: checkpoint自动压缩
# 启用zstd压缩
sudo mount -o remount,compress=zstd ~/mnt/btrfs

# 训练时保存checkpoint
python train.py --checkpoint-dir ~/mnt/btrfs/checkpoints
# checkpoint文件自动压缩，节省~50-70%空间

# 场景2: 共享相同数据（CoW特性）
# 多个实验共享同一数据集
sudo btrfs subvolume snapshot ~/mnt/btrfs/base ~/mnt/btrfs/exp1
sudo btrfs subvolume snapshot ~/mnt/btrfs/base ~/mnt/btrfs/exp2
# exp1和exp2共享数据，只存储差异部分
```

## 完整工作流示例

### 训练实验管理流程

```bash
#!/bin/bash
# train_experiment.sh

EXP_NAME="exp_lr0.001_batch32"
BASE_DIR=~/mnt/btrfs

# 1. 创建基线快照
sudo btrfs subvolume snapshot -r $BASE_DIR $BASE_DIR/../snapshots/baseline_$(date +%Y%m%d)

# 2. 创建实验子卷
sudo btrfs subvolume create $BASE_DIR/$EXP_NAME

# 3. 复制代码到实验子卷（CoW，快速）
cp -r $BASE_DIR/APT-Transformer $BASE_DIR/$EXP_NAME/

# 4. 修改配置
sed -i 's/lr: .*/lr: 0.001/' $BASE_DIR/$EXP_NAME/config.yaml
sed -i 's/batch_size: .*/batch_size: 32/' $BASE_DIR/$EXP_NAME/config.yaml

# 5. 训练前快照
sudo btrfs subvolume snapshot $BASE_DIR/$EXP_NAME $BASE_DIR/../snapshots/${EXP_NAME}_before

# 6. 运行训练
cd $BASE_DIR/$EXP_NAME/APT-Transformer
python train.py

# 7. 训练后快照
sudo btrfs subvolume snapshot -r $BASE_DIR/$EXP_NAME $BASE_DIR/../snapshots/${EXP_NAME}_after

# 8. 增量同步到备份
sudo btrfs send -p $BASE_DIR/../snapshots/baseline_$(date +%Y%m%d) \
                      $BASE_DIR/../snapshots/${EXP_NAME}_after | \
  ssh backup "sudo btrfs receive /backup/experiments/"

echo "Experiment $EXP_NAME completed!"
```

### A/B测试工作流

```bash
#!/bin/bash
# ab_test.sh

MODEL_A=~/mnt/btrfs/exp_model_a
MODEL_B=~/mnt/btrfs/exp_model_b

# 创建两个实验环境
sudo btrfs subvolume create $MODEL_A
sudo btrfs subvolume create $MODEL_B

# CoW复制代码（瞬间完成）
cp -r ~/APT-Transformer $MODEL_A/
cp -r ~/APT-Transformer $MODEL_B/

# 配置不同参数
echo "learning_rate: 0.001" > $MODEL_A/config.yaml
echo "learning_rate: 0.0001" > $MODEL_B/config.yaml

# 并行训练（使用两个终端）
# Terminal 1:
cd $MODEL_A && python train.py &

# Terminal 2:
cd $MODEL_B && python train.py &

# 对比结果
echo "Model A loss:"
cat $MODEL_A/checkpoints/final/metrics.json

echo "Model B loss:"
cat $MODEL_B/checkpoints/final/metrics.json
```

## 监控与维护

### 查看btrfs状态

```bash
# 查看空间使用
btrfs filesystem df ~/mnt/btrfs
btrfs filesystem usage ~/mnt/btrfs

# 查看设备信息
btrfs filesystem show

# 查看子卷列表
sudo btrfs subvolume list -a ~/mnt/btrfs

# 查看快照列表
sudo btrfs subvolume list -s ~/mnt/btrfs

# 查看IO统计
sudo btrfs device stats ~/mnt/btrfs
```

### 定期维护

```bash
# 1. 清理旧的快照（释放空间）
sudo btrfs subvolume delete ~/snapshots/exp_old_*

# 2. 压缩检查
sudo btrfs filesystem defrag -r ~/mnt/btrfs/checkpoints/

# 3. 平衡数据分布
sudo btrfs balance start -dusage=75 ~/mnt/btrfs

# 4. 检查文件系统
sudo btrfs scrub start ~/mnt/btrfs
sudo btrfs scrub status ~/mnt/btrfs
```

## 与真实集群的对比

| **特性** | **btrfs实现** | **集群环境** | **优势** |
|---------|--------------|-------------|---------|
| Checkpoint | 快照 | 定期保存到共享存储 | 本地回滚毫秒级 |
| 数据同步 | send/receive | rsync/NFS分发 | 增量同步，带宽少 |
| 环境隔离 | 子卷 | 容器/虚拟机 | 轻量级，秒级切换 |
| 空间效率 | CoW+压缩 | Dedup系统 | 透明，无需配置 |
| 并行实验 | 多子卷 | 多计算节点 | 本地并行开发 |

## 最佳实践

### 1. 快照命名规范

```bash
# 使用时间戳和描述
snapshots/baseline_20250223_170000
snapshots/exp_lr0.001_after_20250223_180000
snapshots/checkpoint_epoch50_20250223_190000
```

### 2. 定期清理策略

```bash
# 保留最近7天的快照
find ~/snapshots -mtime +7 -exec sudo btrfs subvolume delete {} \;

# 或保留最近N个快照
ls -t ~/snappoints | tail -n +10 | xargs -I {} sudo btrfs subvolume delete ~/snappoints/{}
```

### 3. 压缩策略

```bash
# 对于checkpoint：启用压缩
sudo btrfs property set ~/mnt/btrfs/checkpoints compression zstd

# 对于日志：不压缩（写入频繁）
sudo btrfs property set ~/mnt/btrfs/logs compression none

# 对于代码：启用压缩（节省空间）
sudo btrfs property set ~/mnt/btrfs/APT-Transformer compression zstd
```

### 4. 容量规划

```bash
# 定期检查空间使用
watch -n 60 'df -h ~/mnt/btrfs'

# 设置告警（剩余<20%）
[ $(df ~/mnt/btrfs | tail -1 | awk '{print $5}' | sed 's/%//') -gt 80 ] && \
  echo "WARNING: btrfs almost full!" | mail -s "Alert" admin@example.com
```

## 故障恢复

### 恢复误删文件

```bash
# 1. 找到包含文件的快照
sudo btrfs subvolume list -s ~/mnt/btrfs

# 2. 从快照恢复
sudo btrfs subvolume snapshot ~/snapshots/baseline_safe ~/mnt/btrfs
```

### 修复损坏的文件系统

```bash
# 1. 卸载
sudo umount ~/mnt/btrfs

# 2. 修复
sudo btrfs check --repair ~/btrfs_image/disk.img

# 3. 重新挂载
sudo mount -o loop ~/btrfs_image/disk.img ~/mnt/btrfs
```

## 参考资源

- [Btrfs Wiki](https://btrfs.wiki.kernel.org/)
- [Btrfs Man Pages](https://man7.org/linux/man-pages/man8/btrfs.8.html)
- [Ubuntu Btrfs Documentation](https://ubuntu.com/blog/btrfs)
- [Arch Linux Btrfs Guide](https://wiki.archlinux.org/title/Btrfs)
