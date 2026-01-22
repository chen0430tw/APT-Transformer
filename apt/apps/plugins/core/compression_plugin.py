#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型压缩插件

集成多种压缩技术:
1. 模型剪枝 (Pruning) - 移除不重要的权重
2. 模型量化 (Quantization) - 降低权重精度
3. 知识蒸馏 (Distillation) - 将知识转移到小模型
4. DBC加速训练 (Dimension-Balanced Compression) - 训练加速
5. 低秩分解 (Low-Rank Decomposition) - 权重矩阵分解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
from datetime import datetime


class CompressionPlugin:
    """
    统一的模型压缩插件

    提供多种压缩技术的统一接口
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.name = "apt-compression"
        self.version = "1.0.0"
        self.config = config or {}

        # 压缩配置
        self.compression_methods = self.config.get('methods', ['pruning'])
        self.target_compression_ratio = self.config.get('compression_ratio', 0.5)

        # 各方法的配置
        self.pruning_config = self.config.get('pruning', {
            'ratio': 0.3,
            'type': 'magnitude',
            'structured': False
        })

        self.quantization_config = self.config.get('quantization', {
            'bits': 8,
            'type': 'dynamic',  # static/dynamic/qat
            'backend': 'fbgemm'
        })

        self.distillation_config = self.config.get('distillation', {
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3
        })

        self.dbc_config = self.config.get('dbc', {
            'enabled': True,
            'rank_ratio': 0.1,
            'apply_to_gradients': True
        })

    # ========================================================================
    # 1. 模型剪枝 (Pruning)
    # ========================================================================

    def prune_model(
        self,
        model: nn.Module,
        prune_ratio: float = None,
        prune_type: str = 'magnitude',
        structured: bool = False
    ) -> nn.Module:
        """
        剪枝模型以减少参数量

        Args:
            model: 待剪枝模型
            prune_ratio: 剪枝比例 (0-1)
            prune_type: 剪枝类型 ('magnitude', 'random', 'l1')
            structured: 是否结构化剪枝

        Returns:
            剪枝后的模型
        """
        if prune_ratio is None:
            prune_ratio = self.pruning_config['ratio']

        print(f"✂️  开始模型剪枝 (比例: {prune_ratio*100:.1f}%, 类型: {prune_type})")

        # 收集需要剪枝的参数
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        # 应用剪枝
        if structured:
            # 结构化剪枝 (剪除整个通道/神经元)
            for module, param_name in parameters_to_prune:
                prune.ln_structured(
                    module, name=param_name,
                    amount=prune_ratio, n=2, dim=0
                )
        else:
            # 非结构化剪枝
            if prune_type == 'magnitude':
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=prune_ratio,
                )
            elif prune_type == 'random':
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.RandomUnstructured,
                    amount=prune_ratio,
                )

        # 统计剪枝效果
        total_params = sum(p.numel() for p in model.parameters())
        pruned_params = self._count_pruned_params(model)
        actual_ratio = pruned_params / total_params

        print(f"✅ 剪枝完成! 剪除参数: {pruned_params}/{total_params} ({actual_ratio*100:.2f}%)")

        return model

    def make_pruning_permanent(self, model: nn.Module) -> nn.Module:
        """永久应用剪枝 (移除掩码)"""
        print("🔧 永久应用剪枝...")

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass

        print("✅ 剪枝已永久应用!")
        return model

    def _count_pruned_params(self, model: nn.Module) -> int:
        """计算被剪枝的参数数量"""
        pruned = 0

        for module in model.modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                pruned += (weight == 0).sum().item()

        return pruned

    # ========================================================================
    # 2. 模型量化 (Quantization)
    # ========================================================================

    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: str = None,
        bits: int = None
    ) -> nn.Module:
        """
        量化模型以降低精度和内存占用

        Args:
            model: 待量化模型
            quantization_type: 量化类型 ('dynamic', 'static', 'qat')
            bits: 量化位数 (4, 8, 16)

        Returns:
            量化后的模型
        """
        if quantization_type is None:
            quantization_type = self.quantization_config['type']
        if bits is None:
            bits = self.quantization_config['bits']

        print(f"🔢 开始模型量化 (类型: {quantization_type}, 位数: {bits}bits)")

        model.eval()

        if quantization_type == 'dynamic':
            # 动态量化 (运行时量化)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )

        elif quantization_type == 'static':
            # 静态量化 (需要校准数据)
            model.qconfig = torch.quantization.get_default_qconfig(
                self.quantization_config.get('backend', 'fbgemm')
            )
            torch.quantization.prepare(model, inplace=True)
            # 这里需要用户提供校准数据进行前向传播
            # calibrate(model, calibration_data)
            quantized_model = torch.quantization.convert(model, inplace=False)

        elif quantization_type == 'qat':
            # 量化感知训练 (Quantization-Aware Training)
            model.qconfig = torch.quantization.get_default_qat_qconfig(
                self.quantization_config.get('backend', 'fbgemm')
            )
            quantized_model = torch.quantization.prepare_qat(model, inplace=False)
            # 需要继续训练: train(quantized_model, ...)
            # 最后转换: quantized_model = torch.quantization.convert(quantized_model)

        else:
            raise ValueError(f"未知的量化类型: {quantization_type}")

        # 计算模型大小减少
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = quantized_size / original_size

        print(f"✅ 量化完成! 模型大小: {original_size:.2f}MB → {quantized_size:.2f}MB "
              f"(压缩比: {compression_ratio:.2%})")

        return quantized_model

    def quantize_to_int8(self, model: nn.Module) -> nn.Module:
        """快捷方法: INT8量化"""
        return self.quantize_model(model, quantization_type='dynamic', bits=8)

    def _get_model_size(self, model: nn.Module) -> float:
        """计算模型大小 (MB)"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    # ========================================================================
    # 3. 知识蒸馏 (Distillation)
    # ========================================================================

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        temperature: float = None
    ) -> torch.Tensor:
        """
        知识蒸馏损失

        Args:
            student_logits: 学生模型输出 [batch, seq_len, vocab]
            teacher_logits: 教师模型输出 [batch, seq_len, vocab]
            labels: 真实标签 (可选)
            temperature: 蒸馏温度

        Returns:
            蒸馏损失
        """
        if temperature is None:
            temperature = self.distillation_config['temperature']

        alpha = self.distillation_config['alpha']
        beta = self.distillation_config['beta']

        # 温度软化
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # KL散度损失
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)

        # 如果有真实标签,结合交叉熵损失
        if labels is not None:
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss = alpha * distill_loss + beta * ce_loss
            return total_loss

        return distill_loss

    def train_with_distillation(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 3,
        device: str = 'cuda'
    ):
        """
        使用知识蒸馏训练学生模型

        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            dataloader: 训练数据
            optimizer: 优化器
            num_epochs: 训练轮数
            device: 设备
        """
        print(f"🎓 开始知识蒸馏训练 ({num_epochs} epochs)...")

        student_model.to(device)
        teacher_model.to(device)
        teacher_model.eval()

        for epoch in range(num_epochs):
            student_model.train()
            epoch_loss = 0

            for batch_idx, batch in enumerate(dataloader):
                # 准备数据
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    input_ids = batch.get('input_ids')
                    labels = batch.get('labels', input_ids)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                    labels = batch[1].to(device) if len(batch) > 1 else input_ids
                else:
                    input_ids = batch.to(device)
                    labels = input_ids

                # 教师模型推理
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids)
                    teacher_logits = teacher_outputs if isinstance(teacher_outputs, torch.Tensor) else teacher_outputs[0]

                # 学生模型训练
                student_outputs = student_model(input_ids)
                student_logits = student_outputs if isinstance(student_outputs, torch.Tensor) else student_outputs[0]

                # 计算蒸馏损失
                loss = self.distillation_loss(student_logits, teacher_logits, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(dataloader)
            print(f"📊 Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

        print("✅ 知识蒸馏完成!")

    # ========================================================================
    # 4. DBC加速训练 (Dimension-Balanced Compression)
    # ========================================================================

    def enable_dbc_training(
        self,
        model: nn.Module,
        rank_ratio: float = None,
        apply_to_gradients: bool = True
    ) -> Tuple[nn.Module, Any]:
        """
        启用DBC加速训练

        DBC通过维度平衡压缩来稳定训练和加速收敛

        Args:
            model: APT模型
            rank_ratio: 低秩比例
            apply_to_gradients: 是否应用于梯度稳定

        Returns:
            (模型, DBC优化器)
        """
        if rank_ratio is None:
            rank_ratio = self.dbc_config['rank_ratio']

        print(f"🚀 启用DBC加速训练 (rank_ratio={rank_ratio})")

        # 导入DBC优化器
        from apt_model.modeling.apt_model import DBCDAC_Optimizer, add_gradient_hooks_to_model

        # 创建DBC优化器
        dbc_optimizer = DBCDAC_Optimizer(
            rank_ratio_proj=rank_ratio,
            rank_ratio_res=rank_ratio * 0.5,
            threshold=1e-6,
            iterations=1,
            use_quantization=False,
            quant_bits=8,
            apply_to_gradients=apply_to_gradients
        )

        # 为模型添加梯度稳定钩子
        if apply_to_gradients:
            hooks = add_gradient_hooks_to_model(model, dbc_optimizer)
            print(f"✅ DBC梯度稳定钩子已添加 ({len(hooks)} 个参数)")

        print("✅ DBC加速训练已启用!")

        return model, dbc_optimizer

    # ========================================================================
    # 5. 低秩分解 (Low-Rank Decomposition)
    # ========================================================================

    def low_rank_decomposition(
        self,
        model: nn.Module,
        rank_ratio: float = 0.5,
        layer_types: tuple = (nn.Linear,)
    ) -> nn.Module:
        """
        对模型权重进行低秩分解

        将 W (m×n) 分解为 U (m×r) @ V (r×n)，其中 r << min(m,n)

        Args:
            model: 待分解模型
            rank_ratio: 秩比例 (0-1)
            layer_types: 要分解的层类型

        Returns:
            分解后的模型
        """
        print(f"📊 开始低秩分解 (rank_ratio={rank_ratio})")

        decomposed_layers = 0
        original_params = sum(p.numel() for p in model.parameters())

        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    m, n = weight.shape

                    # 计算目标秩
                    rank = max(1, int(min(m, n) * rank_ratio))

                    # SVD分解
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

                    # 保留前r个奇异值
                    U_r = U[:, :rank]
                    S_r = S[:rank]
                    Vh_r = Vh[:rank, :]

                    # 重构权重
                    weight_approx = U_r @ torch.diag(S_r) @ Vh_r
                    module.weight.data = weight_approx

                    decomposed_layers += 1

        new_params = sum(p.numel() for p in model.parameters())
        compression_ratio = new_params / original_params

        print(f"✅ 低秩分解完成! 分解层数: {decomposed_layers}")
        print(f"   参数量: {original_params} → {new_params} (压缩比: {compression_ratio:.2%})")

        return model

    # ========================================================================
    # 6. 综合压缩流程
    # ========================================================================

    def compress_model(
        self,
        model: nn.Module,
        methods: List[str] = None,
        target_ratio: float = None
    ) -> Dict[str, Any]:
        """
        综合压缩模型

        Args:
            model: 待压缩模型
            methods: 压缩方法列表 ['pruning', 'quantization', 'low_rank']
            target_ratio: 目标压缩比

        Returns:
            压缩结果字典
        """
        if methods is None:
            methods = self.compression_methods
        if target_ratio is None:
            target_ratio = self.target_compression_ratio

        print("="*60)
        print("🗜️  开始综合模型压缩")
        print("="*60)

        # 记录原始模型信息
        original_size = self._get_model_size(model)
        original_params = sum(p.numel() for p in model.parameters())

        results = {
            'original_size_mb': original_size,
            'original_params': original_params,
            'methods_applied': [],
            'compression_stages': []
        }

        # 应用各种压缩方法
        if 'pruning' in methods:
            print("\n📍 步骤 1: 模型剪枝")
            model = self.prune_model(model)
            model = self.make_pruning_permanent(model)

            current_params = sum(p.numel() for p in model.parameters())
            results['methods_applied'].append('pruning')
            results['compression_stages'].append({
                'method': 'pruning',
                'params': current_params,
                'compression_ratio': current_params / original_params
            })

        if 'low_rank' in methods:
            print("\n📍 步骤 2: 低秩分解")
            model = self.low_rank_decomposition(model)

            current_params = sum(p.numel() for p in model.parameters())
            results['methods_applied'].append('low_rank')
            results['compression_stages'].append({
                'method': 'low_rank',
                'params': current_params,
                'compression_ratio': current_params / original_params
            })

        if 'quantization' in methods:
            print("\n📍 步骤 3: 模型量化")
            model = self.quantize_model(model)

            results['methods_applied'].append('quantization')
            results['compression_stages'].append({
                'method': 'quantization',
                'size_mb': self._get_model_size(model)
            })

        # 最终统计
        final_size = self._get_model_size(model)
        final_params = sum(p.numel() for p in model.parameters() if not p.dtype.is_floating_point or p.dtype == torch.float32)

        results['final_size_mb'] = final_size
        results['final_params'] = final_params
        results['size_compression_ratio'] = final_size / original_size
        results['param_compression_ratio'] = final_params / original_params

        print("\n" + "="*60)
        print("📊 压缩结果总结")
        print("="*60)
        print(f"原始模型: {original_size:.2f}MB, {original_params:,} 参数")
        print(f"压缩后:   {final_size:.2f}MB, {final_params:,} 参数")
        print(f"压缩比:   {results['size_compression_ratio']:.2%} (大小), "
              f"{results['param_compression_ratio']:.2%} (参数)")
        print(f"应用方法: {', '.join(results['methods_applied'])}")
        print("="*60)

        return results

    # ========================================================================
    # 7. 压缩评估
    # ========================================================================

    def evaluate_compression(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        test_dataloader: torch.utils.data.DataLoader = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        评估压缩效果

        Args:
            original_model: 原始模型
            compressed_model: 压缩后模型
            test_dataloader: 测试数据
            device: 设备

        Returns:
            评估结果
        """
        print("📊 评估压缩效果...")

        results = {}

        # 1. 模型大小对比
        original_size = self._get_model_size(original_model)
        compressed_size = self._get_model_size(compressed_model)

        results['size_reduction'] = {
            'original_mb': original_size,
            'compressed_mb': compressed_size,
            'reduction_ratio': compressed_size / original_size,
            'saved_mb': original_size - compressed_size
        }

        # 2. 参数量对比
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())

        results['param_reduction'] = {
            'original': original_params,
            'compressed': compressed_params,
            'reduction_ratio': compressed_params / original_params,
            'saved': original_params - compressed_params
        }

        # 3. 推理速度对比 (如果提供了测试数据)
        if test_dataloader is not None:
            import time

            original_model.to(device)
            compressed_model.to(device)

            # 预热
            with torch.no_grad():
                for i, batch in enumerate(test_dataloader):
                    if i >= 2:
                        break
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()}
                        original_model(**batch)
                        compressed_model(**batch)
                    else:
                        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                        original_model(inputs)
                        compressed_model(inputs)

            # 测试原始模型速度
            start_time = time.time()
            with torch.no_grad():
                for i, batch in enumerate(test_dataloader):
                    if i >= 10:
                        break
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()}
                        original_model(**batch)
                    else:
                        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                        original_model(inputs)
            original_time = time.time() - start_time

            # 测试压缩模型速度
            start_time = time.time()
            with torch.no_grad():
                for i, batch in enumerate(test_dataloader):
                    if i >= 10:
                        break
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()}
                        compressed_model(**batch)
                    else:
                        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                        compressed_model(inputs)
            compressed_time = time.time() - start_time

            results['inference_speed'] = {
                'original_time_sec': original_time,
                'compressed_time_sec': compressed_time,
                'speedup': original_time / compressed_time
            }

        # 打印结果
        print("\n压缩评估结果:")
        print(f"  大小: {original_size:.2f}MB → {compressed_size:.2f}MB "
              f"({results['size_reduction']['reduction_ratio']:.2%})")
        print(f"  参数: {original_params:,} → {compressed_params:,} "
              f"({results['param_reduction']['reduction_ratio']:.2%})")

        if 'inference_speed' in results:
            print(f"  推理速度: {results['inference_speed']['speedup']:.2f}x 加速")

        return results

    # ========================================================================
    # 8. 🔮 WebUI/API导出接口
    # ========================================================================

    def export_for_webui(self, export_path: str = None) -> Dict[str, Any]:
        """
        导出压缩数据供WebUI/API使用

        未来API端点:
        - POST /api/compress/prune - 剪枝模型
        - POST /api/compress/quantize - 量化模型
        - POST /api/compress/distill - 知识蒸馏
        - POST /api/compress/full - 综合压缩
        - GET /api/compress/evaluate - 评估压缩效果

        Args:
            export_path: JSON文件导出路径（可选）

        Returns:
            压缩配置和统计数据
        """
        data = {
            'plugin_info': {
                'name': self.name,
                'version': self.version
            },
            'compression_config': {
                'methods': self.compression_methods,
                'target_ratio': self.target_compression_ratio,
                'pruning': self.pruning_config,
                'quantization': self.quantization_config,
                'distillation': self.distillation_config,
                'dbc': self.dbc_config
            },
            'available_methods': [
                {
                    'name': 'pruning',
                    'description': '模型剪枝 - 移除不重要的权重',
                    'typical_compression': '30-50%',
                    'speed_impact': '轻微提升'
                },
                {
                    'name': 'quantization',
                    'description': '模型量化 - 降低权重精度',
                    'typical_compression': '75% (INT8)',
                    'speed_impact': '2-4x加速'
                },
                {
                    'name': 'distillation',
                    'description': '知识蒸馏 - 转移到小模型',
                    'typical_compression': '50-90%',
                    'speed_impact': '视模型大小'
                },
                {
                    'name': 'dbc',
                    'description': 'DBC加速训练 - 维度平衡压缩',
                    'typical_compression': '训练加速',
                    'speed_impact': '训练加速20-30%'
                },
                {
                    'name': 'low_rank',
                    'description': '低秩分解 - 权重矩阵分解',
                    'typical_compression': '40-60%',
                    'speed_impact': '中等提升'
                }
            ],
            'generated_at': datetime.now().isoformat()
        }

        # 导出到JSON文件
        if export_path:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    def generate_compression_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """
        生成压缩报告 (Markdown格式)

        Args:
            results: 压缩结果
            output_path: 报告保存路径

        Returns:
            Markdown格式报告
        """
        report = []
        report.append("# 模型压缩报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 压缩方法
        report.append("## 应用的压缩方法\n")
        for method in results.get('methods_applied', []):
            report.append(f"- ✅ {method}")

        # 压缩效果
        report.append("\n## 压缩效果\n")
        report.append(f"- **原始模型大小**: {results['original_size_mb']:.2f} MB")
        report.append(f"- **压缩后大小**: {results['final_size_mb']:.2f} MB")
        report.append(f"- **大小压缩比**: {results['size_compression_ratio']:.2%}")
        report.append(f"- **原始参数量**: {results['original_params']:,}")
        report.append(f"- **压缩后参数量**: {results['final_params']:,}")
        report.append(f"- **参数压缩比**: {results['param_compression_ratio']:.2%}")

        # 各阶段详情
        if 'compression_stages' in results:
            report.append("\n## 压缩阶段详情\n")
            for stage in results['compression_stages']:
                method = stage['method']
                if 'compression_ratio' in stage:
                    report.append(f"- **{method}**: 压缩比 {stage['compression_ratio']:.2%}")

        report_text = '\n'.join(report)

        # 保存报告
        if output_path:
            report_file = Path(output_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("APT模型压缩插件示例\n")

    # 配置
    config = {
        'methods': ['pruning', 'quantization'],
        'compression_ratio': 0.5,
        'pruning': {
            'ratio': 0.3,
            'type': 'magnitude'
        },
        'quantization': {
            'bits': 8,
            'type': 'dynamic'
        }
    }

    plugin = CompressionPlugin(config)

    # 创建测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    model = SimpleModel()

    print(f"原始模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"原始模型大小: {plugin._get_model_size(model):.2f} MB\n")

    # 综合压缩
    results = plugin.compress_model(model, methods=['pruning', 'low_rank'])

    # 生成报告
    report = plugin.generate_compression_report(results)
    print("\n" + report)

    # 导出WebUI数据
    webui_data = plugin.export_for_webui()
    print(f"\n🔮 WebUI数据已导出: {len(webui_data['available_methods'])} 种压缩方法可用")

    print("\n✅ 模型压缩插件示例完成!")
