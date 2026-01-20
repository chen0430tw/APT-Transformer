"""
虚拟Blackwell PyTorch集成模块

将虚拟Blackwell优化无缝集成到APT模型训练中。

注意：此模块需要PyTorch。
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from typing import Optional, Dict
    from apt_model.optimization.virtual_blackwell_adapter import create_virtual_blackwell


    class VBOptimizedLinear(nn.Module):
        """使用虚拟Blackwell优化的线性层（GPU加速版）"""

        def __init__(self, in_features: int, out_features: int,
                     mode: str = 'auto', bias: bool = True,
                     enable_quantization: bool = True):
            """
            Args:
                in_features: 输入维度
                out_features: 输出维度
                mode: 'auto', 'training', 'inference', 'precision'
                bias: 是否使用bias
                enable_quantization: 是否启用量化
            """
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

            # 虚拟Blackwell适配器
            self.vb_adapter = create_virtual_blackwell(
                mode=mode,
                enable_quantization=enable_quantization
            )
            self.layer_id = f'linear_{id(self)}'
            self._registered = False

        def forward(self, x):
            """前向传播（GPU加速，无CPU转换）"""
            # 注册权重（仅首次）- 直接使用tensor
            if not self._registered:
                self.vb_adapter.register_weight(self.layer_id, self.weight.detach())
                self._registered = True

            # 获取权重和输入 - 保持在GPU上
            W = self.weight
            X = x

            # 处理维度 (batch, seq, dim) -> (dim, batch*seq)
            original_shape = X.shape
            if len(original_shape) == 3:
                batch, seq, dim = original_shape
                X = X.reshape(batch * seq, dim).T
            elif len(original_shape) == 2:
                X = X.T
            else:
                raise ValueError(f"Unsupported input shape: {original_shape}")

            # 虚拟Blackwell压缩计算 - 全程在GPU上
            Y = self.vb_adapter.compress(W, X, self.layer_id)

            # 转置回来
            Y = Y.T

            # 恢复维度
            if len(original_shape) == 3:
                Y = Y.reshape(batch, seq, -1)

            # 添加bias - 确保在同一设备上
            if self.bias is not None:
                Y = Y + self.bias.to(Y.device)

            return Y

        def get_stats(self) -> Dict:
            """获取虚拟Blackwell统计信息"""
            return self.vb_adapter.get_stats()

        def print_stats(self):
            """打印统计信息"""
            self.vb_adapter.print_stats()


    class VBModelWrapper(nn.Module):
        """将现有模型的线性层替换为VB优化版本"""

        def __init__(self, model: nn.Module, mode: str = 'auto',
                     enable_quantization: bool = True,
                     replace_pattern: str = 'all'):
            """
            Args:
                model: 原始模型
                mode: VB模式
                enable_quantization: 是否启用量化
                replace_pattern: 替换模式 ('all', 'large', 'custom')
            """
            super().__init__()
            self.model = model
            self.mode = mode
            self.enable_quantization = enable_quantization
            self.replaced_layers = []

            # 替换线性层
            if replace_pattern == 'all':
                self._replace_all_linear()
            elif replace_pattern == 'large':
                self._replace_large_linear()

        def _replace_all_linear(self):
            """替换所有线性层"""
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and not isinstance(module, VBOptimizedLinear):
                    self._replace_module(name, module)

        def _replace_large_linear(self, threshold: int = 512):
            """只替换大型线性层"""
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and not isinstance(module, VBOptimizedLinear):
                    if module.in_features >= threshold or module.out_features >= threshold:
                        self._replace_module(name, module)

        def _replace_module(self, name: str, module: nn.Linear):
            """替换单个模块"""
            vb_linear = VBOptimizedLinear(
                module.in_features,
                module.out_features,
                mode=self.mode,
                bias=module.bias is not None,
                enable_quantization=self.enable_quantization
            )

            # 复制权重
            vb_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                vb_linear.bias.data.copy_(module.bias.data)

            # 替换
            parent_name, attr_name = self._get_parent_and_attr(name)
            if parent_name:
                parent = dict(self.model.named_modules())[parent_name]
                setattr(parent, attr_name, vb_linear)
            else:
                setattr(self.model, attr_name, vb_linear)

            self.replaced_layers.append(name)
            print(f"✓ 替换层: {name} ({module.in_features} -> {module.out_features})")

        def _get_parent_and_attr(self, name: str):
            """获取父模块和属性名"""
            parts = name.split('.')
            if len(parts) == 1:
                return None, parts[0]
            return '.'.join(parts[:-1]), parts[-1]

        def forward(self, *args, **kwargs):
            """前向传播"""
            return self.model(*args, **kwargs)

        def get_all_stats(self) -> Dict:
            """获取所有VB层的统计信息"""
            stats = {}
            for name, module in self.model.named_modules():
                if isinstance(module, VBOptimizedLinear):
                    stats[name] = module.get_stats()
            return stats

        def print_all_stats(self):
            """打印所有统计信息"""
            print("\n" + "="*70)
            print("虚拟Blackwell优化统计 - 全局汇总")
            print("="*70)

            all_stats = self.get_all_stats()

            # 汇总
            total_gpu_hits = 0
            total_accesses = 0
            total_svd_saves = 0

            for name, stats in all_stats.items():
                if 'layer1_vgpu' in stats:
                    vgpu = stats['layer1_vgpu']
                    total_gpu_hits += vgpu.get('gpu_hits', 0)
                    total_accesses += vgpu.get('total', 0)

                if 'layer2_microvm' in stats:
                    microvm = stats['layer2_microvm']
                    total_svd_saves += microvm.get('hits', 0)

            print(f"\n已优化层数: {len(self.replaced_layers)}")
            print(f"总GPU命中: {total_gpu_hits}/{total_accesses} " +
                  f"({total_gpu_hits/total_accesses*100:.1f}%)" if total_accesses > 0 else "")
            print(f"总SVD节省: {total_svd_saves}次")

            print("\n详细统计:")
            for name in self.replaced_layers[:5]:  # 只显示前5个
                module = dict(self.model.named_modules())[name]
                if isinstance(module, VBOptimizedLinear):
                    print(f"\n[{name}]")
                    module.print_stats()

            if len(self.replaced_layers) > 5:
                print(f"\n... 还有 {len(self.replaced_layers)-5} 个优化层未显示")

            print("="*70 + "\n")


    def enable_vb_optimization(model: nn.Module, mode: str = 'training',
                              enable_quantization: bool = True,
                              replace_pattern: str = 'all') -> VBModelWrapper:
        """
        快速启用虚拟Blackwell优化

        Args:
            model: 要优化的模型
            mode: 'auto', 'training', 'inference', 'precision'
            enable_quantization: 是否启用量化
            replace_pattern: 'all' 或 'large'

        Returns:
            包装后的模型

        Example:
            model = APTLargeModel(config)
            model = enable_vb_optimization(model, mode='training')
        """
        print("\n" + "="*70)
        print("启用虚拟Blackwell优化")
        print("="*70)
        print(f"模式: {mode}")
        print(f"量化: {'启用' if enable_quantization else '禁用'}")
        print(f"替换策略: {replace_pattern}")
        print()

        wrapper = VBModelWrapper(
            model,
            mode=mode,
            enable_quantization=enable_quantization,
            replace_pattern=replace_pattern
        )

        print(f"\n✅ 成功替换 {len(wrapper.replaced_layers)} 个线性层")
        print("="*70 + "\n")

        return wrapper

else:
    # PyTorch不可用时的占位符
    VBOptimizedLinear = None
    VBModelWrapper = None
    enable_vb_optimization = None
