# WSL2 9P协议技术分析

> 研究日期: 2026-02-23
> 研究背景: Virtual VRAM测试中遇到WSL2文件I/O性能问题

## 1. 9P协议架构

### 完整调用链路

```
Linux应用 → Linux VFS → virtio-9p客户端 → 9P消息 → VMBus → 9P服务器(Windows) → Windows NT API → NTFS
```

### 关键组件

| **组件** | **位置** | **作用** |
|---------|---------|---------|
| `virtio-9p` | Linux内核 | 9P客户端，将文件系统调用转换为9P消息 |
| `9pfs.sys` | Windows内核 | 9P服务器，接收9P消息并转换为NT API调用 |
| VMBus | Hyper-V | 虚拟机总线，Windows和WSL2之间的通信通道 |
| ext4.vhdx | C盘 | WSL2虚拟磁盘，存储Linux原生文件系统 |

### 性能数据对比

| **存储位置** | **IOPS (随机4K)** | **延迟** | **吞吐** |
|-------------|-----------------|---------|---------|
| `/home/` (ext4) | ~15,000 IOPS | 低 | ~1.0 GB/s |
| `/mnt/d/` (9P) | ~2,000 IOPS | 高 | ~12-50 MB/s |

**性能损失**: 9P协议导致约 **7.5倍IOPS下降** 和 **20-80倍吞吐下降**

### 9P挂载参数

```bash
$ mount | grep 9p
# 典型输出：
# drvfs on /mnt/c type 9p (rw,noatime,dirsync,access=client,cache=mmap,trans=virtio,version=9p2000.L)
```

## 2. 为什么选择9P协议

### WSL1 vs WSL2架构差异

| **特性** | **WSL1** | **WSL2** |
|---------|---------|---------|
| 架构 | 翻译层 | 轻量级VM (Hyper-V) |
| 文件访问 | DrvFS直接访问NTFS | 需要跨VM协议 |
| 内核 | 无真实Linux内核 | 完整Linux内核 5.15+ |
| 兼容性 | 有限（系统调用翻译） | 几乎完整POSIX |

### 9P vs 其他方案对比

| **方案** | **优点** | **缺点** | **为何不适合** |
|---------|---------|---------|-------------|
| **9P** | 内核原生支持、协议简单、virtio集成 | 性能有开销 | ✅ **选择** |
| **SMB** | 成熟、Windows原生 | 协议复杂、开销大 | 太重，延迟高 |
| **NFS** | Unix标准、性能较好 | 需要网络服务、复杂 | 虚拟化场景不理想 |
| **SSHFS** | 安全、易部署 | SSH加密开销大 | 延迟最高 |

### 为什么选9P：技术原因

1. **Linux内核原生支持**
   ```bash
   # 9P客户端直接编译进内核
   CONFIG_9P_FS=y
   CONFIG_NET_9P=y
   CONFIG_VIRTIO_9P=y
   ```
   不需要额外用户空间服务

2. **协议设计简单**
   - 只有13种基本消息类型
   - 无需认证、加密（本地VM不需要）
   - 状态less设计，适合虚拟化

3. **与virtio完美集成**
   ```bash
   mount -t 9p -o trans=virtio,version=9p2000.L hostfs /mnt/host
   ```

4. **开发维护成本低**（成熟技术）

## 3. virtio-fs + DAX：9P的替代方案

### 性能对比数据

| **测试项目** | **virtio-fs** | **9P协议** | **本地磁盘** |
|------------|-------------|-----------|------------|
| **文件读取 (1GB)** | 2.1秒 | 4.8秒 | 1.8秒 |
| **文件写入 (1GB)** | 2.4秒 | 5.2秒 | 2.0秒 |
| **小文件操作 (1000个)** | 3.2秒 | 8.9秒 | 2.8秒 |
| **项目编译** | 45秒 | 78秒 | 38秒 |

**结论**: virtio-fs比9P快 **2-3倍**，接近本地磁盘性能

### DAX (Direct Access) 原理

**架构对比**：

```
传统9P路径：
应用 → VFS → 9P客户端 → 序列化 → VMBus → 9P服务器 → NT API → NTFS

virtio-fs with DAX路径：
应用 → VFS → DAX窗口 → 直接映射宿主机页缓存 → NTFS
    零拷贝访问
```

**DAX性能数据**：

| **指标** | **无DAX** | **有DAX** | **提升** |
|---------|----------|----------|---------|
| **4K随机读IOPS** | ~8k | **12k** | +50% |
| **4K随机读带宽** | ~30 MB/s | **49 MB/s** | +63% |
| **内存占用** | +300-350MB | 几乎为0 | -100% |
| **VM退出次数** | 高频 | 极低 | -90%+ |

### virtio-fs完整架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Linux Guest (VM)                          │
│  应用层 → VFS层 → FUSE客户端 → virtio-fs驱动                    │
└─────────────────────────────────────────────────────────────────┘
                    ↓ virtio队列 (共享内存)
┌─────────────────────────────────────────────────────────────────┐
│                      QEMU / Hypervisor                           │
│  vhost-user-fs-pci设备 → DAX窗口 (2GB共享内存)                   │
└─────────────────────────────────────────────────────────────────┘
                    ↓ vhost-user协议
┌─────────────────────────────────────────────────────────────────┐
│                      Host机器                                    │
│  virtiofsd守护进程 → 文件系统 → 页缓存                          │
└─────────────────────────────────────────────────────────────────┘
```

### WSL2现状

**没有发现WSL2计划迁移到virtio-fs的官方消息**

可能原因：
1. **历史包袱**: WSL2于2019年发布，virtio-fs当时还不成熟
2. **Hyper-V限制**: WSL2使用Hyper-V而非QEMU/KVM，virtio-fs主要针对KVM优化
3. **Windows实现复杂度**: 需要在Windows侧实现virtiofsd，工程量大

## 4. 与Virtual VRAM的相似性

### 本质相同：边界跨越开销

| **维度** | **Virtual VRAM** | **WSL2 9P** |
|---------|-----------------|-------------|
| **快速存储** | GPU显存 (8GB) | Linux ext4 (`/home`) |
| **慢速存储** | CPU内存 (DDR) | Windows NTFS (`/mnt/d`) |
| **访问路径** | GPU → PCIe → CPU | Linux → 9P → Windows |
| **协议开销** | PCIe事务层 | 9P消息序列化 |
| **性能损失** | ~10-50倍 | ~2-80倍 |

### 对应的优化策略

| **Virtual VRAM** | **WSL2 9P** | **原理** |
|-----------------|-------------|---------|
| **prefetch** | **文件复制到/home** | 预先搬到快速存储 |
| **Arc nested** | **无对应** | 分层存储管理 |
| **pinned memory优化** | **9P缓存优化** | 减少传输开销 |
| **零拷贝 (D2D)** | **DAX (virtio-fs)** | 零拷贝访问 |

### 通用原则

```
分层存储系统 → 边界跨越开销 → 需要智能数据放置
```

类似案例：
- CPU缓存 → RAM → 磁盘
- CDN边缘节点 → 源站
- 分布式系统的本地缓存 → 远程存储

## 5. 最佳实践建议

### WSL2文件系统使用策略

1. **代码和工作文件放在WSL2内部**
   ```bash
   # ✅ 推荐
   ~/APT-Transformer/        # 项目主目录
   ~/datasets/               # 热数据集

   # ❌ 避免
   /mnt/d/APT-Transformer/   # 通过9P访问慢
   ```

2. **大数据集的访问策略**
   ```bash
   # 选项1: 软链接（访问时才用9P）
   ln -s /mnt/d/datasets/large_dataset ~/datasets/large

   # 选项2: 只复制热数据
   cp -r /mnt/d/datasets/small_dataset ~/datasets/

   # 选项3: 定期同步
   rsync -av /mnt/d/datasets/ ~/datasets/
   ```

3. **避免的操作**
   ```bash
   # ❌ 不要频繁执行
   git status /mnt/d/project          # 慢
   npm install /mnt/d/node_modules    # 慢
   find /mnt/d -name "*.py"           # 慢

   # ✅ 改为WSL2内部
   cd ~/project && git status         # 快
   ```

### C盘空间管理

1. **WSL2虚拟磁盘位置**
   ```
   C:\Users\asus\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx
   ```

2. **清理WSL2空间**
   ```bash
   # 在WSL2内清理
   sudo apt clean
   rm -rf ~/.cache/*

   # 在Windows压缩虚拟磁盘
   wsl --shutdown
   Optimize-VHD -Path "C:\...\ext4.vhdx" -Mode Full
   ```

3. **迁移WSL2到其他盘**
   ```bash
   # 导出
   wsl --export Ubuntu D:\WSL2\ubuntu.tar

   # 导入（指定新位置）
   wsl --import Ubuntu D:\WSL2\ D:\WSL2\ubuntu.tar
   ```

## 6. 调试命令

### 检查9P挂载
```bash
mount | grep 9p
df -h | grep drvfs
```

### 监控I/O性能
```bash
iostat -x 1        # 总体I/O统计
iotop -o           # 按进程查看I/O
```

### 检查文件系统性能
```bash
# 顺序读写
dd if=/dev/zero of=~/test.img bs=1G count=1 oflag=direct
dd if=~/test.img of=/dev/null bs=1G count=1 iflag=direct

# 随机读写
fio --name=randread --ioengine=libaio --iodepth=1 --rw=randread --bs=4k --direct=1 --size=1G --numjobs=4 --filename=~/test
```

## 7. 参考资料

- [比较 WSL 版本 - Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/compare-versions)
- [Virtio-fs Overview](https://m.blog.csdn.net/Rong_Toa/article/details/115286564)
- [智汇华云 \| kata container virtiofs测试和技术分析](https://m.blog.csdn.net/weilidai/article/details/118758870)
- [揭秘VSCode+WSL2文件访问缓慢真相：9P协议性能调优的5个关键步骤](https://m.blog.csdn.net/codetrick/article/details/155202823)
- [仅限内部流传的WSL2调优方法：让VSCode远程开发体验媲美原生Linux](https://m.blog.csdn.net/simcode/article/details/154874316)
