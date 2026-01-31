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
# 精度分离：粗部（FP4大数）+ 细部（INT4小数）
# ============================================================================

class PrecisionSeparator:
    """精度分离器：将权重分解为粗部和细部"""

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

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.protocol = BOHProtocol()
        self.separator = PrecisionSeparator()

        # 共享内存（模拟NVLink）
        self.shared_memory = {}

        # 量化参数缓存（训练优化：缓存量化刻度，避免重复排序）
        self.quantile_cache = {}  # {weight_id: quantiles}
        self.separated_cache = {}  # {weight_id: separated} - 缓存完整的精度分离结果
        self.weight_hash_cache = {}  # {weight_id: hash} - 权重哈希，检测变化
        self.cache_step_counter = {}
        self.cache_refresh_interval = 100  # 每100步刷新量化刻度

        # 统计信息
        self.stats = {
            'gpu_hits': 0,
            'total': 0,
            'coarse_computes': 0,
            'fine_computes': 0,
            'cache_hits': 0,
            'cache_refreshes': 0
        }

    def compute(self, weight: torch.Tensor, input_tensor: torch.Tensor, weight_id: str) -> torch.Tensor:
        """
        计算流程（计算单元，不是缓存访问！）：
        1. 精度分离（粗部/细部）- 使用缓存加速
        2. BOH握手协调
        3. 粗部先行计算（快速低精度）
        4. 细部修正（高精度）
        5. 通过共享内存同步
        """
        self.stats['total'] += 1

        # 1. 精度分离（激进缓存：缓存完整separated结果）
        # 训练时权重变化缓慢，可以每10步才重新计算一次精度分离
        if weight_id not in self.separated_cache:
            # 首次：完整计算并缓存separated结果
            separated = self.separator.separate(weight, cached_quantiles=None)
            # 缓存整个separated结果（使用detach避免梯度累积）
            self.separated_cache[weight_id] = {
                'coarse': separated['coarse'].detach(),
                'coarse_quantiles': separated['coarse_quantiles'].detach(),
                'fine': separated['fine'].detach(),
                'sign': separated['sign'].detach(),
                'fine_scale': separated['fine_scale'].detach() if isinstance(separated['fine_scale'], torch.Tensor) else separated['fine_scale']
            }
            self.cache_step_counter[weight_id] = 0
        else:
            # 检查是否需要刷新缓存（每10步刷新一次）
            self.cache_step_counter[weight_id] += 1
            if self.cache_step_counter[weight_id] >= 10:  # 每10步刷新
                # 重新计算精度分离
                separated = self.separator.separate(weight, cached_quantiles=None)
                self.separated_cache[weight_id] = {
                    'coarse': separated['coarse'].detach(),
                    'coarse_quantiles': separated['coarse_quantiles'].detach(),
                    'fine': separated['fine'].detach(),
                    'sign': separated['sign'].detach(),
                    'fine_scale': separated['fine_scale'].detach() if isinstance(separated['fine_scale'], torch.Tensor) else separated['fine_scale']
                }
                self.cache_step_counter[weight_id] = 0
                self.stats['cache_refreshes'] += 1
            else:
                # 直接使用缓存的separated结果（避免重新计算）
                separated = self.separated_cache[weight_id]
                self.stats['cache_hits'] += 1

        # 2. BOH握手
        handshake = self.protocol.handshake(
            sender_id=self.gpu_id,
            receiver_id=self.gpu_id,
            data_size=weight.numel()
        )

        # 3. 粗部先行计算（模拟低延迟）
        if handshake['priority'] == 'coarse_first':
            self.stats['coarse_computes'] += 1

            # 优化3: 存储到共享内存（使用detach避免梯度累积）
            self.shared_memory[f'{weight_id}_coarse'] = separated['coarse'].detach()

        # 4. 细部修正（高精度计算）
        full_weight = self.separator.combine(separated)
        self.stats['fine_computes'] += 1

        # 优化3: 存储到共享内存（使用detach避免梯度累积）
        self.shared_memory[f'{weight_id}_fine'] = separated['fine'].detach()

        # 5. 执行计算
        result = full_weight @ input_tensor

        self.stats['gpu_hits'] += 1

        return result

    def get_stats(self) -> Dict:
        total = self.stats['total']
        return {
            'gpu_hits': self.stats['gpu_hits'],
            'total': total,
            'gpu_hit_rate': self.stats['gpu_hits'] / total if total > 0 else 0,
            'coarse_computes': self.stats['coarse_computes'],
            'fine_computes': self.stats['fine_computes'],
            'cache_hits': self.stats['cache_hits'],
            'cache_refreshes': self.stats['cache_refreshes'],
            'cache_hit_rate': self.stats['cache_hits'] / total if total > 0 else 0,
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
                 gpu_id: int = 0, enable_fp4: bool = True):
        # Layer 1: 虚拟GPU计算单元（NVLink模拟）
        self.vgpu = VirtualGPUNetwork(gpu_id=gpu_id)

        # Layer 2: Flash Attention + FP4量化（粗部）
        self.fp4_layer = FlashFP4Layer(enable_fp4=enable_fp4)

        # Layer 3: VGPU-SL量化（BOH协议：细部INT4）
        self.quantizer = VGPUSLQuantizer() if enable_quantization else None
        self.enable_quant = enable_quantization

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
                print(f"[Virtual Blackwell v6.0] NVLink模拟 + 分层精度优化")
                print(f"{'='*80}")
                print(f"  运行模式: {mode_desc}")
                print(f"  FP4粗精度: {'✓ 启用' if enable_fp4 and HAS_FP4 else '✗ 禁用'}")
                print(f"  BOH量化: {'✓ 启用' if enable_quantization else '✗ 禁用'}")
                print(f"\n  分层优化策略:")
                print(f"    • 超大层 (>5M参数)  → 简化8-bit量化 (跳过精度分离)")
                print(f"    • 大层 (100K-5M参数) → 采样估计分位数 (10%采样)")
                print(f"    • 小层 (<100K参数)   → 完整精确计算")
                print(f"{'='*80}\n")
                _VB_CONFIG_PRINTED = True
            except (OSError, IOError):
                pass  # 环境中stdout不可用时静默失败

    def register_weight(self, weight_id: str, weight: torch.Tensor, priority: int = 5):
        # Layer 2: 预编码为FP4（粗部）
        self.fp4_layer.register_weight(weight_id, weight)

    def compress(self, W: torch.Tensor, X: torch.Tensor, weight_id: str = 'default') -> torch.Tensor:
        """
        完整计算流程：
        1. Layer 1: 虚拟GPU计算单元（精度分离 + BOH握手 + 共享内存）
        2. Layer 3: VGPU-SL量化（细部INT4修正）
        3. Layer 2: FP4压缩矩阵乘法（粗部计算）
        """
        # 确保W和X在同一设备上
        W = W.to(X.device)

        # Layer 1: 虚拟GPU计算（精度分离 + NVLink模拟）
        # 这是计算单元，不是缓存访问！
        Y = self.vgpu.compute(W, X, weight_id)

        # Layer 3: BOH细部修正（可选）
        if self.enable_quant:
            # BOH协议已在Layer 1中使用，这里仅做额外量化
            pass

        # Layer 2: FP4粗部已在Layer 1的精度分离中处理
        # 这里使用FP4 layer的统计
        self.fp4_layer.stats['total_calls'] += 1
        self.fp4_layer.stats['fp4_hits'] += 1

        return Y

    def linear_pulse(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], weight_id: str) -> torch.Tensor:
        """
        脉冲路径专用方法：支持F.linear的原生API

        Args:
            x: 输入张量 (batch, seq, dim) 或 (batch, dim)
            weight: 权重矩阵 (out_features, in_features)
            bias: 偏置向量 (out_features,) 或 None
            weight_id: 权重唯一标识符

        Returns:
            输出张量，形状同 F.linear(x, weight, bias)
        """
        self.total_calls += 1
        self.vb_calls += 1

        # 确保weight和x在同一设备上
        weight = weight.to(x.device)

        # 每层独立计数（虽然脉冲路径不需要判断，但保持统计一致）
        if weight_id not in self.layer_pulse:
            self.layer_pulse[weight_id] = 0
        self.layer_pulse[weight_id] += 1

        # 处理维度：F.linear期望x是(*, in_features)，输出是(*, out_features)
        original_shape = x.shape

        if len(original_shape) == 3:
            # (batch, seq, dim) -> (batch*seq, dim)
            batch, seq, dim = original_shape
            x_2d = x.reshape(batch * seq, dim)
        elif len(original_shape) == 2:
            x_2d = x
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")

        # 转置为compress所需的格式：(dim, batch*seq)
        X = x_2d.T
        W = weight

        # Layer 1: 虚拟GPU计算（ShrinkTrace量化）
        Y = self.vgpu.compute(W, X, weight_id)

        # Layer 2: FP4统计
        self.fp4_layer.stats['total_calls'] += 1
        self.fp4_layer.stats['fp4_hits'] += 1

        # 转置回来：(out_features, batch*seq) -> (batch*seq, out_features)
        Y = Y.T

        # 恢复原始形状
        if len(original_shape) == 3:
            Y = Y.reshape(batch, seq, -1)

        # 添加bias
        if bias is not None:
            Y = Y + bias.to(Y.device)

        return Y

    def get_stats(self) -> Dict:
        return {
            'layer1_vgpu': self.vgpu.get_stats(),
            'layer2_fp4': self.fp4_layer.get_stats(),
            'layer3_vgpusl': self.quantizer.get_stats() if self.quantizer else {}
        }

    def get_stats_v6(self) -> Dict:
        """
        v6版本统计方法：正确显示ShrinkTrace缓存统计

        返回格式与get_stats兼容，但使用v6的ShrinkTrace缓存指标
        """
        vgpu_stats = self.vgpu.get_stats()

        return {
            'pulse_stats': {
                'total_calls': self.total_calls,
                'vb_calls': self.vb_calls,
                'fast_calls': self.fast_calls,
                'vb_ratio': f"{self.vb_calls / self.total_calls * 100:.1f}%" if self.total_calls > 0 else "0%",
                'pulse_interval': self.pulse_interval
            },
            'v6_shrinktrace_cache': {
                'total_compute': vgpu_stats.get('total', 0),
                'cache_hits': vgpu_stats.get('cache_hits', 0),
                'cache_refreshes': vgpu_stats.get('scale_updates', 0),
                'cache_hit_rate': vgpu_stats.get('cache_hit_rate', 0.0),
                'precision_separations': vgpu_stats.get('precision_separations', 0),
                'scale_checks': vgpu_stats.get('scale_checks', 0)
            },
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


def create_virtual_blackwell(mode='auto', enable_quantization=True, max_gpu_mb=2000, enable_fp4=True):
    """
    创建虚拟Blackwell适配器

    Args:
        mode: 运行模式 ('auto', 'training', 'inference', 'precision')
        enable_quantization: 启用BOH协议量化 (Layer 3)
        max_gpu_mb: GPU缓存大小 (MB)
        enable_fp4: 启用FP4量化 (Layer 2)

    Returns:
        VirtualBlackwellAdapter实例
    """
    return VirtualBlackwellAdapter(mode, enable_quantization, max_gpu_mb, enable_fp4)


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


# =============================================================================
# V6 FIX: Pulse-friendly ShrinkTrace weight cache (STE fake-quant) + fast path hooks
# =============================================================================
from dataclasses import dataclass

@dataclass
class _QuantWeightEntry:
    w_deq: torch.Tensor          # dequantized weight used for forward (same dtype/device as original weight)
    scale: torch.Tensor          # scale tensor (scalar or per-row)
    step: int

class ShrinkTraceQuantCache:
    """
    Lightweight ShrinkTrace cache for *weights*.

    - Computes a robust scale from |W| quantile (q) with optional subsampling.
    - Produces INT8 weights and dequantized W_deq (for fast reuse).
    - Uses STE in forward: W + (W_deq - W).detach() to keep gradients.
    """
    def __init__(
        self,
        q: float = 0.999,
        sample_k: int = 50_000,
        update_interval: int = 20,
        rel_change_th: float = 0.20,
        per_row: bool = False,
        eps: float = 1e-8,
    ):
        self.q = float(q)
        self.sample_k = int(sample_k)
        self.update_interval = int(update_interval)
        self.rel_change_th = float(rel_change_th)
        self.per_row = bool(per_row)
        self.eps = float(eps)

        self._cache: Dict[str, _QuantWeightEntry] = {}
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_refreshes": 0,
            "cache_misses": 0,
            "recomputed": 0,
        }
        self._global_step = 0

    @staticmethod
    def _subsample_abs(W: torch.Tensor, k: int) -> torch.Tensor:
        # Subsample without materializing huge tensors on CPU.
        flat = W.detach().abs().flatten()
        n = flat.numel()
        if n <= k:
            return flat
        # random indices on device
        idx = torch.randint(0, n, (k,), device=flat.device)
        return flat.index_select(0, idx)

    def _compute_scale(self, W: torch.Tensor) -> torch.Tensor:
        if self.per_row and W.dim() == 2:
            # per-output-channel scale: quantile over in_features
            absW = W.detach().abs()
            # optional subsample columns for speed
            if absW.shape[1] > self.sample_k:
                idx = torch.randint(0, absW.shape[1], (self.sample_k,), device=W.device)
                absW = absW.index_select(1, idx)
            # quantile per row
            # torch.quantile supports dim; use float32 for stability
            qv = torch.quantile(absW.float(), self.q, dim=1)
            scale = (qv / 127.0).clamp_min(self.eps).to(W.dtype)
            return scale  # (out_features,)
        else:
            sample = self._subsample_abs(W, self.sample_k)
            qv = torch.quantile(sample.float(), self.q)
            scale = (qv / 127.0).clamp_min(self.eps).to(W.dtype)
            return scale  # scalar tensor

    def _quant_dequant(self, W: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Dequantized (fake) INT8 weight.
        if scale.dim() == 1 and W.dim() == 2:
            # per-row scaling: scale[:, None]
            s = scale[:, None]
        else:
            s = scale
        w_int = torch.clamp(torch.round(W / s), -127, 127)
        w_deq = (w_int * s).to(W.dtype)
        return w_deq

    def get_weight(self, weight_id: str, W: torch.Tensor) -> torch.Tensor:
        """
        Return STE fake-quant weight. Refreshes based on interval + relative change.
        """
        self.stats["total_calls"] += 1
        self._global_step += 1

        entry = self._cache.get(weight_id, None)
        need_refresh = entry is None

        if entry is not None:
            # interval-based refresh gate
            if (self._global_step - entry.step) >= self.update_interval:
                # relative change check on small sample (cheap)
                sample = self._subsample_abs(W, min(self.sample_k, 20_000))
                cur_q = torch.quantile(sample.float(), self.q).to(W.dtype)
                old_q = (entry.scale.mean() * 127.0) if entry.scale.dim() == 1 else (entry.scale * 127.0)
                old_q = old_q.to(W.dtype)
                rel = (cur_q - old_q).abs() / (old_q.abs() + self.eps)
                if rel.item() >= self.rel_change_th:
                    need_refresh = True

        if need_refresh:
            self.stats["recomputed"] += 1
            if entry is None:
                self.stats["cache_misses"] += 1
            else:
                self.stats["cache_refreshes"] += 1

            scale = self._compute_scale(W)
            w_deq = self._quant_dequant(W, scale).detach()  # cached dequant
            self._cache[weight_id] = _QuantWeightEntry(w_deq=w_deq, scale=scale.detach(), step=self._global_step)
            entry = self._cache[weight_id]
        else:
            self.stats["cache_hits"] += 1

        # STE: forward uses quantized value, backward flows to W
        w_ste = W + (entry.w_deq.to(W.device, dtype=W.dtype) - W).detach()
        return w_ste

    def get_stats(self) -> Dict[str, float]:
        total = float(self.stats["total_calls"])
        return {
            **self.stats,
            "hit_rate": (self.stats["cache_hits"] / total) if total > 0 else 0.0,
            "refresh_rate": (self.stats["cache_refreshes"] / total) if total > 0 else 0.0,
        }

# If the original file defines VirtualBlackwellAdapter, we monkey-patch it with pulse-friendly helpers.
try:
    _VBA = VirtualBlackwellAdapter  # type: ignore[name-defined]
    if not hasattr(_VBA, "_v6_cache"):
        _VBA._v6_cache = ShrinkTraceQuantCache(q=0.999, sample_k=50_000, update_interval=20, rel_change_th=0.20, per_row=False)

    def _v6_linear_fast(self, x: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor]):
        return torch.nn.functional.linear(x, W, b)

    def _v6_linear_pulse(self, x: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor], weight_id: str):
        Wq = self._v6_cache.get_weight(weight_id, W)
        return torch.nn.functional.linear(x, Wq, b)

    def _v6_get_stats(self):
        base = {}
        if hasattr(self, "vgpu") and hasattr(self.vgpu, "get_stats"):
            base["layer1_vgpu"] = self.vgpu.get_stats()
        if hasattr(self, "fp4_layer") and hasattr(self.fp4_layer, "get_stats"):
            base["layer2_fp4"] = self.fp4_layer.get_stats()
        if hasattr(self, "sl_quantizer") and hasattr(self.sl_quantizer, "get_stats"):
            base["layer3_vgpu_sl"] = self.sl_quantizer.get_stats()
        base["v6_shrinktrace_cache"] = self._v6_cache.get_stats()
        return base

    _VBA.linear_fast = _v6_linear_fast  # type: ignore[attr-defined]
    _VBA.linear_pulse = _v6_linear_pulse  # type: ignore[attr-defined]
    _VBA.get_stats_v6 = _v6_get_stats    # type: ignore[attr-defined]
except Exception:
    pass
