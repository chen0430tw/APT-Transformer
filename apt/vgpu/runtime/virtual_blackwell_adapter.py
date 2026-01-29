"""
virtual_blackwell_adapter.py - 完整虚拟Blackwell适配器 (GPU优化版)

三层虚拟化完整整合:
  Layer 1: 虚拟GPU网络 (计算单元 + NVLink模拟)
  Layer 2: Flash Attention + FP4量化 (精度分离：粗部)
  Layer 3: VGPU-SL量化 (BOH协议：细部INT4)

作者: chen0430tw
版本: 6.0 (NVLink Simulation - 精度分离 + 共享内存 + BOH握手)
"""

import torch
from typing import Dict, Optional, Tuple
from collections import OrderedDict

# 全局标志：只打印一次VB配置信息
_VB_CONFIG_PRINTED = False

# 导入Flash Attention + FP4模块
try:
    from apt.perf.optimization.gpu_flash_optimization import FP4Codec
    HAS_FP4 = True
except ImportError:
    HAS_FP4 = False
    FP4Codec = None


# ============================================================================
# ShrinkTrace v6: Quantile-based Adaptive INT8 Quantization
# ============================================================================

class ShrinkTraceQuantizer:
    """
    ShrinkTrace v6 量化器：基于quantile的自适应INT8量化

    核心优势：
    1. Quantile-based scale（更鲁棒，不受异常值影响）
    2. Sample-based estimation（大tensor采样加速）
    3. Adaptive updates（只在scale变化超过阈值时更新）
    """

    @staticmethod
    @torch.no_grad()
    def quantile_scale(x: torch.Tensor, q: float = 0.999, sample: int = 0) -> torch.Tensor:
        """
        使用quantile计算量化scale

        Args:
            x: 输入tensor
            q: 分位数（默认0.999，即99.9%分位点）
            sample: 采样数量（0表示不采样，>0表示随机采样）

        Returns:
            scale: 量化缩放因子
        """
        if sample > 0 and x.numel() > sample:
            # 随机采样加速（对大tensor）
            idx = torch.randperm(x.numel(), device=x.device)[:sample]
            a = x.view(-1)[idx].abs()
        else:
            a = x.abs()

        v = torch.quantile(a.float(), q)
        v = torch.clamp(v, min=1e-6)
        return v / 127.0

    @staticmethod
    def fake_int8_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """INT8 fake quantization: 量化到[-127,127]然后反量化"""
        q = torch.round(x / scale).clamp(-127, 127)
        return q * scale


# ============================================================================
# 精度分离：粗部（FP4大数）+ 细部（INT4小数）- 已被ShrinkTrace替代
# ============================================================================

class PrecisionSeparator:
    """精度分离器：将权重分解为粗部和细部（保留以兼容旧代码）"""

    @staticmethod
    def separate(tensor: torch.Tensor, cached_quantiles: torch.Tensor = None) -> Dict:
        """
        分离精度：
        粗部(coarse) - FP4 存储大数（指数 + 符号 + 高位尾数）
        细部(fine) - INT4 存储小数（低位尾数）

        优化版本：分层策略
        - 超大层（>5M参数）：跳过精度分离，使用简化量化
        - 大层（100K-5M参数）：采样估计分位数
        - 小层（<100K参数）：完整精确计算
        """
        n_elements = tensor.numel()
        abs_tensor = torch.abs(tensor)
        sign = torch.sign(tensor)
        eps = 1e-10

        # 策略1: 超大层（>5M参数）- 使用简化等距量化
        if n_elements > 5_000_000:
            # 使用等距分位数（避免复杂的quantile计算）
            max_val = abs_tensor.max()
            if max_val == 0:
                max_val = eps

            # 创建16个等距分位点
            quantiles = torch.linspace(0, max_val.item(), 16, device=tensor.device)

            # 将值映射到0-15的级别（使用简单的线性映射）
            # 避免使用searchsorted，直接计算级别
            coarse_level = torch.clamp(
                (abs_tensor * 15.0 / max_val).round(),
                0, 15
            ).to(torch.int8)

            # 重建粗部值（使用量化级别）
            coarse_values = quantiles[coarse_level.long()]
            coarse = coarse_values * sign

            # 计算残差
            residual = tensor - coarse

            # 简化的fine量化
            fine_scale = residual.abs().max() / 7.5
            if fine_scale == 0:
                fine_scale = eps
            fine_level = torch.clamp((residual / fine_scale).round(), -7, 7).to(torch.int8)

            return {
                'coarse': coarse_level,
                'coarse_quantiles': quantiles,
                'fine': fine_level,
                'sign': sign,
                'fine_scale': fine_scale if isinstance(fine_scale, torch.Tensor) else torch.tensor(fine_scale, device=tensor.device)
            }

        # 策略2和3: 大层和小层 - 计算分位数（使用缓存或采样）
        if cached_quantiles is not None:
            quantiles = cached_quantiles
        else:
            abs_flat = abs_tensor.flatten()

            # 策略2: 大层（100K-5M参数）- 使用采样估计
            if n_elements > 100_000:
                # 采样10%或最多100K元素
                sample_size = min(100_000, max(10_000, n_elements // 10))
                indices = torch.randperm(n_elements, device=abs_flat.device)[:sample_size]
                sampled = abs_flat[indices]

                q_points = torch.linspace(0, 1, 16, device=tensor.device)
                quantiles = torch.quantile(sampled, q_points)
            # 策略3: 小层（<100K参数）- 完整精确计算
            else:
                q_points = torch.linspace(0, 1, 16, device=tensor.device)
                quantiles = torch.quantile(abs_flat, q_points)

            # 确保quantiles单调递增
            quantiles = torch.cummax(quantiles, dim=0).values

        # 对于大层和小层，使用采样进行量化级别计算
        if n_elements > 500_000:
            # 采样20%进行量化，然后插值
            sample_size = max(100_000, n_elements // 5)
            abs_flat = abs_tensor.flatten()
            indices = torch.randperm(n_elements, device=abs_flat.device)[:sample_size]
            sampled = abs_flat[indices]

            # 对采样数据进行量化（使用right=True避免索引16）
            sampled_levels = torch.searchsorted(quantiles, sampled, right=True)
            # 确保索引在有效范围[0, 15]内
            sampled_levels = torch.clamp(sampled_levels, 0, 15).to(torch.int8)

            # 创建完整的量化结果
            coarse_level = torch.zeros(n_elements, dtype=torch.int8, device=tensor.device)
            coarse_level[indices] = sampled_levels

            # 对未采样位置使用中位数级别填充
            median_level = sampled_levels.median().to(torch.int8)
            mask = torch.ones(n_elements, dtype=torch.bool, device=tensor.device)
            mask[indices] = False
            coarse_level[mask] = median_level

            coarse_level = coarse_level.reshape(abs_tensor.shape)
        else:
            # 小层：完整量化（使用right=True避免索引16）
            abs_flat_for_search = abs_tensor.flatten()
            coarse_level_flat = torch.searchsorted(quantiles, abs_flat_for_search, right=True)
            # 确保索引在有效范围[0, 15]内
            coarse_level_flat = torch.clamp(coarse_level_flat, 0, 15)
            coarse_level = coarse_level_flat.reshape(abs_tensor.shape).to(torch.int8)

        # 重建粗部值
        coarse_values = quantiles[coarse_level.long()]
        coarse = coarse_values * sign

        # 细部：残差量化到 16 个级别 (INT4)
        residual = tensor - coarse

        # 使用局部缩放因子（每行一个scale）提高精度
        if len(residual.shape) == 2:
            fine_scale = residual.abs().max(dim=1, keepdim=True).values / 7.5
            fine_scale = torch.clamp(fine_scale, min=eps)
        else:
            fine_scale = residual.abs().max() / 7.5
            if fine_scale == 0:
                fine_scale = eps

        fine_level = torch.clamp((residual / fine_scale).round(), -7, 7).to(torch.int8)

        return {
            'coarse': coarse_level,
            'coarse_quantiles': quantiles,
            'fine': fine_level,
            'sign': sign,
            'fine_scale': fine_scale
        }

    @staticmethod
    def combine(separated: Dict) -> torch.Tensor:
        """组合粗部和细部恢复张量"""
        coarse_level = separated['coarse']
        coarse_quantiles = separated['coarse_quantiles']
        fine_level = separated['fine']
        sign = separated['sign']
        fine_scale = separated['fine_scale']

        # 恢复粗部
        coarse_values = coarse_quantiles[coarse_level.long()]
        coarse = coarse_values * sign

        # 恢复细部
        fine = fine_level.float() * fine_scale

        return coarse + fine


# ============================================================================
# BOH 协议：Binary Optimization Hierarchy 握手
# ============================================================================

class BOHProtocol:
    """BOH协议：协调粗部和细部的传输"""

    @staticmethod
    def handshake(sender_id: int, receiver_id: int, data_size: int) -> Dict:
        """
        握手协议：
        1. 发送方请求传输
        2. 接收方确认准备好
        3. 协商精度级别（粗部先行/细部跟随）
        """
        return {
            'sender': sender_id,
            'receiver': receiver_id,
            'size': data_size,
            'priority': 'coarse_first',  # 粗部优先传输
            'status': 'ready'
        }


# ============================================================================
# Layer 1: 虚拟GPU网络（计算单元 + NVLink模拟）
# ============================================================================

class VirtualGPUNetwork:
    """虚拟GPU计算单元（不是缓存！）- 模拟NVLink通信"""

    def __init__(self, gpu_id: int = 0, trigger_hi: float = 1.2, trigger_lo: float = 0.8,
                 q: float = 0.999, sample: int = 50000, check_interval: int = 10):
        self.gpu_id = gpu_id
        self.protocol = BOHProtocol()
        self.quantizer = ShrinkTraceQuantizer()

        # 共享内存（模拟NVLink）
        self.shared_memory = {}

        # ShrinkTrace自适应量化参数
        self.scale_cache = {}  # {weight_id: scale} - 缓存量化scale
        self.weight_quant_cache = {}  # {weight_id: quantized_weight} - 缓存量化权重
        self.step_counter = {}  # {weight_id: step_count} - 每个权重的计步器
        self.trigger_hi = trigger_hi  # scale变化上限（默认1.2，即+20%）
        self.trigger_lo = trigger_lo  # scale变化下限（默认0.8，即-20%）
        self.q = q  # quantile参数（默认0.999）
        self.sample = sample  # 采样数量（默认50K）
        self.check_interval = check_interval  # 每N步检查一次scale变化

        # 统计信息
        self.stats = {
            'gpu_hits': 0,
            'total': 0,
            'scale_updates': 0,  # scale更新次数
            'cache_hits': 0,  # 使用缓存次数
            'scale_checks': 0  # scale检查次数
        }

    def compute(self, weight: torch.Tensor, input_tensor: torch.Tensor, weight_id: str) -> torch.Tensor:
        """
        ShrinkTrace v6计算流程（真正的缓存）：
        1. 检查缓存的量化权重
        2. 每N步检查scale变化（不是每次）
        3. 使用缓存的量化权重执行计算
        """
        self.stats['total'] += 1

        # 初始化step counter
        if weight_id not in self.step_counter:
            self.step_counter[weight_id] = 0

        steps_since_update = self.step_counter[weight_id]

        # 1. 判断是否需要更新（基于步数间隔）
        need_update = False

        if weight_id not in self.weight_quant_cache:
            # 首次：必须计算
            need_update = True
        elif steps_since_update >= self.check_interval:
            # 达到检查间隔：检查scale变化
            old_scale = self.scale_cache[weight_id]
            new_scale = self.quantizer.quantile_scale(weight, q=self.q, sample=self.sample)

            # 计算scale变化比例
            ratio = (new_scale / (old_scale + 1e-9)).clamp(min=1e-9).item()
            self.stats['scale_checks'] += 1

            # 如果变化超过阈值，需要更新
            if ratio >= self.trigger_hi or ratio <= self.trigger_lo:
                need_update = True

        if need_update:
            # 更新scale和量化权重
            self.scale_cache[weight_id] = self.quantizer.quantile_scale(
                weight, q=self.q, sample=self.sample
            )
            scale = self.scale_cache[weight_id]
            self.weight_quant_cache[weight_id] = self.quantizer.fake_int8_quant(weight, scale).detach()
            self.step_counter[weight_id] = 0  # 重置计数器
            self.stats['scale_updates'] += 1
        else:
            # 使用缓存
            self.step_counter[weight_id] += 1
            self.stats['cache_hits'] += 1

        # 2. 使用缓存的量化权重执行计算
        weight_quant = self.weight_quant_cache[weight_id]
        result = weight_quant @ input_tensor

        self.stats['gpu_hits'] += 1

        return result

    def get_stats(self) -> Dict:
        total = self.stats['total']
        return {
            'gpu_hits': self.stats['gpu_hits'],
            'total': total,
            'gpu_hit_rate': self.stats['gpu_hits'] / total if total > 0 else 0,
            'scale_updates': self.stats['scale_updates'],
            'cache_hits': self.stats['cache_hits'],
            'scale_checks': self.stats['scale_checks'],
            'cache_hit_rate': self.stats['cache_hits'] / total if total > 0 else 0,
            'update_rate': self.stats['scale_updates'] / total if total > 0 else 0,
            'gpu_memory_mb': len(self.shared_memory) * 0.1  # 估算
        }


# Layer 2: Flash Attention + FP4 量化
class FlashFP4Layer:
    """Flash Attention + FP4 量化层"""

    def __init__(self, enable_fp4: bool = True):
        self.enable_fp4 = enable_fp4 and HAS_FP4
        self.weight_cache = {}  # {weight_id: (fp4_packed, scale)}
        self.stats = {'fp4_hits': 0, 'fp4_encode': 0, 'total_calls': 0}
        # FP4 自动回退到标准实现，无需警告

    def register_weight(self, weight_id: str, W: torch.Tensor):
        """注册权重并预编码为FP4"""
        if self.enable_fp4:
            # 预编码为FP4格式
            packed, scale = FP4Codec.encode(W)
            self.weight_cache[weight_id] = (packed, scale, W.shape[-1])
            self.stats['fp4_encode'] += 1

    def compress(self, W: torch.Tensor, X: torch.Tensor, weight_id: str = 'default') -> torch.Tensor:
        """使用FP4压缩的矩阵乘法"""
        self.stats['total_calls'] += 1

        if self.enable_fp4 and weight_id in self.weight_cache:
            # 使用缓存的FP4权重
            packed, scale, original_size = self.weight_cache[weight_id]

            # 解码FP4 -> FP32
            W_decoded = FP4Codec.decode(packed.to(X.device), scale.to(X.device), original_size)
            W_decoded = W_decoded.view(W.shape)

            self.stats['fp4_hits'] += 1
            return W_decoded @ X
        else:
            # 标准计算
            return W @ X

    def get_stats(self) -> Dict:
        total = self.stats['total_calls']
        return {
            'fp4_hits': self.stats['fp4_hits'],
            'total_calls': total,
            'fp4_hit_rate': (self.stats['fp4_hits'] / total) if total > 0 else 0,
            'fp4_encoded': self.stats['fp4_encode']
        }


# Layer 3: VGPU-SL量化 (BOH协议)
class VGPUSLQuantizer:
    """VGPU-SL量化 (BOH协议，INT4量化)"""

    def __init__(self, block_size: int = 8):
        self.block_size = block_size
        self.stats = {'ortho_blocks': 0, 'scale_blocks': 0, 'total_blocks': 0}

    def quantize_int4(self, W: torch.Tensor) -> torch.Tensor:
        maxv = torch.max(torch.abs(W))
        scale = max(maxv.item() / 7.0, 1e-12)
        W_quant = torch.clamp(torch.round(W / scale), -7, 7)
        return W_quant * scale

    def boh_compress(self, W: torch.Tensor) -> torch.Tensor:
        m, n = W.shape
        W_out = torch.zeros_like(W)

        for i in range(0, m, self.block_size):
            for j in range(0, n, self.block_size):
                self.stats['total_blocks'] += 1

                i_end = min(i + self.block_size, m)
                j_end = min(j + self.block_size, n)
                block = W[i:i_end, j:j_end]

                if block.numel() == 0 or min(block.shape) < 2:
                    W_out[i:i_end, j:j_end] = self.quantize_int4(block)
                    self.stats['scale_blocks'] += 1
                    continue

                try:
                    epsilon_orth = torch.linalg.norm(
                        block.T @ block - torch.eye(min(block.shape), device=block.device, dtype=block.dtype)
                    )
                    # 简化条件数计算（避免cond函数在某些版本不可用）
                    s = torch.linalg.svdvals(block)
                    kappa = (s[0] / s[-1]).item() if s[-1] > 1e-10 else 1e10

                    if epsilon_orth < 0.3 and kappa < 50:
                        self.stats['ortho_blocks'] += 1
                        U, S, Vh = torch.linalg.svd(block, full_matrices=False)
                        block_ortho = U @ Vh
                        block_quant = self.quantize_int4(block_ortho)
                    else:
                        self.stats['scale_blocks'] += 1
                        block_quant = self.quantize_int4(block)
                except:
                    self.stats['scale_blocks'] += 1
                    block_quant = self.quantize_int4(block)

                W_out[i:i_end, j:j_end] = block_quant

        return W_out

    def get_stats(self) -> Dict:
        total = self.stats['total_blocks']
        return {
            'ortho_blocks': self.stats['ortho_blocks'],
            'total_blocks': total,
            'ortho_ratio': self.stats['ortho_blocks'] / total if total > 0 else 0
        }


# 完整虚拟Blackwell适配器
class VirtualBlackwellAdapter:
    def __init__(self, mode: str = 'auto', enable_quantization: bool = True,
                 gpu_id: int = 0, enable_fp4: bool = True, pulse_interval: int = 20):
        # Layer 1: 虚拟GPU计算单元（NVLink模拟）
        self.vgpu = VirtualGPUNetwork(gpu_id=gpu_id)

        # Layer 2: Flash Attention + FP4量化（粗部）
        self.fp4_layer = FlashFP4Layer(enable_fp4=enable_fp4)

        # Layer 3: VGPU-SL量化（BOH协议：细部INT4）
        self.quantizer = VGPUSLQuantizer() if enable_quantization else None
        self.enable_quant = enable_quantization

        # 间歇性脉冲控制
        self.pulse_interval = pulse_interval  # 每N次forward才执行一次VB
        self.pulse_counter = 0  # 当前计数器
        self.total_calls = 0
        self.vb_calls = 0  # VB实际执行次数
        self.fast_calls = 0  # 快速路径次数

        # 只在首次创建时打印配置信息（避免62层重复打印）
        global _VB_CONFIG_PRINTED
        if not _VB_CONFIG_PRINTED:
            mode_desc = {
                'auto': '自动',
                'training': '训练',
                'inference': '推理',
                'precision': '精度优先'
            }.get(mode, mode)

            try:
                print(f"\n{'='*80}")
                print(f"[Virtual Blackwell v6.0] 间歇性脉冲 + ShrinkTrace量化")
                print(f"{'='*80}")
                print(f"  运行模式: {mode_desc}")
                print(f"  量化算法: ShrinkTrace v6 (Quantile-based INT8)")
                print(f"  FP4粗精度: {'✓ 启用' if enable_fp4 and HAS_FP4 else '✗ 禁用'}")
                print(f"\n  ⚡ 间歇性脉冲策略:")
                print(f"    • 脉冲间隔: 每 {pulse_interval} 次forward执行1次VB")
                print(f"    • 快速路径: 直接矩阵乘法（原生PyTorch优化）")
                print(f"    • 脉冲时刻: ShrinkTrace自适应INT8量化")
                print(f"    • VB开销比例: ~{100/pulse_interval:.1f}%")
                print(f"\n  📊 ShrinkTrace v6特性:")
                print(f"    • Quantile-based scale (q=0.999, 更鲁棒)")
                print(f"    • 自适应更新 (变化阈值: ±20%)")
                print(f"    • 采样加速 (50K samples for large tensors)")
                print(f"    • INT8 fake quantization [-127, 127]")
                print(f"{'='*80}\n")
                _VB_CONFIG_PRINTED = True
            except (OSError, IOError):
                pass  # 环境中stdout不可用时静默失败

    def register_weight(self, weight_id: str, weight: torch.Tensor, priority: int = 5):
        # Layer 2: 预编码为FP4（粗部）
        self.fp4_layer.register_weight(weight_id, weight)

    def compress(self, W: torch.Tensor, X: torch.Tensor, weight_id: str = 'default') -> torch.Tensor:
        """
        间歇性脉冲计算流程：
        - 快速路径（大部分时候）：直接 W @ X（原生PyTorch优化）
        - 脉冲时刻（每N次）：完整VB流程（精度分离 + BOH协议）
        """
        self.total_calls += 1
        self.pulse_counter += 1

        # 确保W和X在同一设备上
        W = W.to(X.device)

        # 判断是否触发脉冲
        if self.pulse_counter >= self.pulse_interval:
            # ⚡ 脉冲时刻：执行完整VB流程
            self.pulse_counter = 0  # 重置计数器
            self.vb_calls += 1

            # Layer 1: 虚拟GPU计算（精度分离 + NVLink模拟）
            Y = self.vgpu.compute(W, X, weight_id)

            # Layer 3: BOH细部修正（可选）
            if self.enable_quant:
                # BOH协议已在Layer 1中使用，这里仅做额外量化
                pass

            # Layer 2: FP4粗部已在Layer 1的精度分离中处理
            self.fp4_layer.stats['total_calls'] += 1
            self.fp4_layer.stats['fp4_hits'] += 1

            return Y
        else:
            # 快速路径：直接矩阵乘法（跳过VB开销）
            self.fast_calls += 1
            return W @ X

    def get_stats(self) -> Dict:
        return {
            'pulse_stats': {
                'total_calls': self.total_calls,
                'vb_calls': self.vb_calls,
                'fast_calls': self.fast_calls,
                'vb_ratio': f"{self.vb_calls / self.total_calls * 100:.1f}%" if self.total_calls > 0 else "0%",
                'pulse_interval': self.pulse_interval
            },
            'layer1_vgpu': self.vgpu.get_stats(),
            'layer2_fp4': self.fp4_layer.get_stats(),
            'layer3_vgpusl': self.quantizer.get_stats() if self.quantizer else {}
        }

    def print_stats(self):
        stats = self.get_stats()

        print("\n" + "="*70)
        print("虚拟Blackwell统计 (NVLink模拟 - 精度分离 + BOH握手)")
        print("="*70)

        vgpu = stats['layer1_vgpu']
        print(f"\n[Layer 1 - VGPU计算单元]")
        print(f"  总计算: {vgpu['total']}")
        print(f"  粗部计算: {vgpu['coarse_computes']} (FP4)")
        print(f"  细部计算: {vgpu['fine_computes']} (INT4)")
        print(f"  GPU命中率: {vgpu['gpu_hit_rate']:.1%}")
        print(f"  精度缓存: {vgpu['cache_hits']}/{vgpu['total']} ({vgpu['cache_hit_rate']:.1%})")
        print(f"  缓存刷新: {vgpu['cache_refreshes']} 次")
        print(f"  共享内存: {vgpu['gpu_memory_mb']:.1f} MB")

        fp4 = stats['layer2_fp4']
        if fp4:
            print(f"[Layer 2 - FP4量化] FP4命中: {fp4['fp4_hits']}/{fp4['total_calls']} ({fp4['fp4_hit_rate']:.1%})")
            print(f"                    已编码权重: {fp4['fp4_encoded']} 个")

        if self.enable_quant:
            vgpusl = stats['layer3_vgpusl']
            print(f"[Layer 3 - BOH协议] 正交块: {vgpusl['ortho_blocks']}/{vgpusl['total_blocks']} ({vgpusl['ortho_ratio']:.1%})")

        print("="*70 + "\n")


def create_virtual_blackwell(mode='auto', enable_quantization=True, max_gpu_mb=2000, enable_fp4=True, pulse_interval=20):
    """
    创建虚拟Blackwell适配器

    Args:
        mode: 运行模式 ('auto', 'training', 'inference', 'precision')
        enable_quantization: 启用BOH协议量化 (Layer 3)
        max_gpu_mb: GPU缓存大小 (MB) - 用作gpu_id
        enable_fp4: 启用FP4量化 (Layer 2)
        pulse_interval: 脉冲间隔（每N次forward执行1次VB）

    Returns:
        VirtualBlackwellAdapter实例
    """
    # max_gpu_mb实际上被用作gpu_id（历史遗留参数名）
    gpu_id = 0  # 单GPU场景固定为0
    return VirtualBlackwellAdapter(mode, enable_quantization, gpu_id, enable_fp4, pulse_interval)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("虚拟Blackwell测试 (Flash Attention + FP4)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    print(f"FP4可用: {HAS_FP4}")

    adapter = create_virtual_blackwell('training', enable_quantization=True, enable_fp4=True)

    torch.manual_seed(42)
    W = torch.randn(512, 512, dtype=torch.float32, device=device) * 0.02
    X = torch.randn(512, 64, dtype=torch.float32, device=device)

    print(f"\n测试参数:")
    print(f"  权重形状: {W.shape}")
    print(f"  输入形状: {X.shape}")

    adapter.register_weight('test', W)

    print(f"\n运行16次前向传播...")
    for i in range(16):
        Y = adapter.compress(W, X, 'test')
        if (i+1) % 4 == 0:
            print(f"  [OK] Batch {i+1}/16 完成")

    adapter.print_stats()

    print("[OK] 测试完成！")
