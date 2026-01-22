#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model training time estimation utilities.
Provides tools to estimate training time based on model configuration, dataset size, and hardware.
"""

import torch
import math
import time
from typing import Dict, Any, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

class TrainingTimeEstimator:
    """
    Training time estimation tool for APT models.
    Provides accurate time estimates based on model configuration,
    dataset properties, and available hardware.
    """
    
    def __init__(self, model_config, dataset_size, batch_size, epochs, logger=None):
        """
        Initialize the training time estimator.
        
        Args:
            model_config: Model configuration object
            dataset_size (int): Number of samples in the dataset
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
            logger (optional): Logger instance for logging messages
        """
        self.logger = logger
        self.model_config = model_config
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Detect hardware configuration
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available and self.gpu_count > 0 else "Unknown"
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if self.gpu_available and self.gpu_count > 0 else 0
        
        # Estimate GPU performance (TFLOPS)
        self.estimated_tflops = self._estimate_gpu_performance()
        
        # CPU information
        self.cpu_cores = psutil.cpu_count(logical=True) if psutil else 0
        self.ram_gb = psutil.virtual_memory().total / (1024**3) if psutil else 0
    
    def _estimate_gpu_performance(self) -> float:
        """
        Estimate GPU computational performance based on the detected GPU model.
        
        Returns:
            float: Estimated performance in TFLOPS (teraFLOPS)
        """
        if not self.gpu_available or self.gpu_count == 0:
            return 0.0
        
        # Mapping of common GPU models to their approximate FP32 performance in TFLOPS
        performance_map = {
            # NVIDIA GeForce RTX 30 series
            "RTX 3090": 35.58,
            "RTX 3080 Ti": 34.1,
            "RTX 3080": 29.77,
            "RTX 3070 Ti": 21.7,
            "RTX 3070": 20.31,
            "RTX 3060 Ti": 16.2,
            "RTX 3060": 12.74,
            
            # NVIDIA GeForce RTX 40 series
            "RTX 4090": 82.58,
            "RTX 4080": 48.74,
            "RTX 4070 Ti": 40.09,
            "RTX 4070": 29.15,
            "RTX 4060 Ti": 22.06,
            "RTX 4060": 15.09,
            
            # NVIDIA GeForce RTX 20 series
            "RTX 2080 Ti": 13.45,
            "RTX 2080": 10.07,
            "RTX 2070": 7.46,
            "RTX 2060": 6.45,
            
            # NVIDIA GeForce GTX series
            "GTX 1080 Ti": 11.34,
            "GTX 1080": 8.87,
            "GTX 1070": 6.46,
            
            # NVIDIA professional GPUs
            "RTX A6000": 38.71,
            "RTX A5000": 27.8,
            "RTX A4000": 19.17,
            "Quadro RTX 8000": 16.3,
            "Quadro RTX 6000": 16.3,
            "Quadro RTX 5000": 11.2,
            "Quadro RTX 4000": 7.1,
            
            # NVIDIA data center GPUs
            "A100": 19.5,  # FP32 (312.0 for Tensor cores with FP16)
            "A100 80GB": 19.5,
            "A100-80GB": 19.5,
            "A100 40GB": 19.5,
            "A800": 19.5,  # Same as A100 but with reduced NVLink bandwidth
            "A30": 24.1,
            "A10": 31.2,
            "A10G": 31.2,
            "H100": 51.0,  # FP32
            "H100 SXM5": 51.0,
            "H100 PCIe": 51.0,
            "H800": 51.0,  # Same as H100 but with reduced NVLink bandwidth
            
            # NVIDIA older data center GPUs
            "V100": 14.0,
            "V100 SXM2": 15.7,
            "V100 PCIe": 14.0,
            "V100-32GB": 15.7,
            "T4": 8.1,
            "P100": 10.6,
            "K80": 8.7,
            "Tesla K80": 8.73,
            "Tesla P4": 5.5,
            "Tesla M40": 7.0,
            "Tesla M60": 9.7,
            
            # NVIDIA Titan series
            "Titan RTX": 16.31,
            "Titan V": 14.9,
            "Titan X": 11.0,
            
            # AMD GPUs
            "Radeon RX 6900 XT": 23.04,
            "Radeon RX 6800 XT": 20.74,
            "Radeon RX 6800": 16.17,
            "Radeon RX 6700 XT": 13.21,
            "Radeon VII": 13.4,
            "MI100": 23.1,
            "MI250": 47.9,
        }
        
        # Try to find an exact match
        for gpu_model, tflops in performance_map.items():
            if gpu_model in self.gpu_name:
                return tflops
        
        # If no exact match, make estimates based on VRAM and GPU family
        if "NVIDIA" in self.gpu_name or "GeForce" in self.gpu_name:
            if "RTX" in self.gpu_name:
                if self.gpu_memory >= 24:  # High-end GPU
                    return 30.0
                elif self.gpu_memory >= 16:  # Mid-high GPU
                    return 20.0
                elif self.gpu_memory >= 8:  # Mid-range GPU
                    return 10.0
                else:  # Entry-level GPU
                    return 5.0
            elif "GTX" in self.gpu_name:
                if self.gpu_memory >= 11:  # High-end GTX
                    return 10.0
                elif self.gpu_memory >= 8:  # Mid-range GTX
                    return 8.0
                else:  # Entry-level GTX
                    return 5.0
            elif "Quadro" in self.gpu_name or "Tesla" in self.gpu_name:
                if self.gpu_memory >= 32:  # High-end professional
                    return 15.0
                elif self.gpu_memory >= 16:  # Mid-range professional
                    return 10.0
                else:  # Entry-level professional
                    return 6.0
        elif "AMD" in self.gpu_name or "Radeon" in self.gpu_name:
            if self.gpu_memory >= 16:  # High-end AMD
                return 20.0
            elif self.gpu_memory >= 8:  # Mid-range AMD
                return 9.0
            else:  # Entry-level AMD
                return 4.0
        
        # Default estimate for unknown GPUs
        return max(self.gpu_memory / 2, 1.0)  # Very rough approximation
    
    def estimate_training_time(self) -> Dict[str, Any]:
        """
        Predict training time based on model configuration, dataset properties, and hardware.
        
        Returns:
            dict: Training time estimates and related information
        """
        # For CPU-only training, provide a rough estimate
        if not self.gpu_available:
            # CPU training is typically 10-50x slower than GPU
            total_hours = self.dataset_size * self.epochs * 0.002  # Very rough estimate
            return {
                "total_seconds": total_hours * 3600,
                "total_hours": total_hours,
                "epoch_hours": total_hours / self.epochs,
                "formatted_total_time": self._format_time(total_hours * 3600),
                "formatted_epoch_time": self._format_time(total_hours * 3600 / self.epochs),
                "steps_per_epoch": self.dataset_size / self.batch_size,
                "total_steps": (self.dataset_size / self.batch_size) * self.epochs,
                "device": "cpu",
                "note": "CPU训练速度显著慢于GPU训练，这只是一个粗略估计。"
            }
        
        # Calculate model size and complexity
        model_complexity = self._calculate_model_complexity()
        params_millions = self._calculate_model_parameters() / 1e6
        
        # Estimate compute requirements per sample based on model complexity
        compute_per_sample = self._calculate_compute_per_sample(model_complexity)
        
        # Adjust for batch size efficiency
        # Larger batch sizes are more efficient up to a point
        batch_efficiency = min(1.5, 0.7 + (0.3 * math.log2(self.batch_size) / math.log2(32)))
        
        # Adjust for GPU performance
        # Higher TFLOPS = faster training
        gpu_performance_factor = 10.0 / max(self.estimated_tflops, 0.1)
        
        # Multi-GPU scaling efficiency
        # Not perfectly linear - communication overhead reduces efficiency
        multi_gpu_efficiency = 0.8  # 80% scaling efficiency
        if self.gpu_count > 1:
            gpu_scaling = 1.0 + (self.gpu_count - 1) * multi_gpu_efficiency
            gpu_performance_factor /= gpu_scaling
        
        # Time per sample in seconds
        estimated_time_per_sample = compute_per_sample * gpu_performance_factor / batch_efficiency
        
        # Calculate total steps
        steps_per_epoch = math.ceil(self.dataset_size / self.batch_size)
        total_steps = steps_per_epoch * self.epochs
        
        # Total compute time
        compute_seconds = total_steps * self.batch_size * estimated_time_per_sample
        
        # Add overhead for data loading, checkpoint saving, evaluation, etc.
        overhead_factor = 1.2  # 20% overhead
        total_seconds = compute_seconds * overhead_factor
        
        # Convert to hours
        total_hours = total_seconds / 3600
        epoch_hours = total_hours / self.epochs
        
        # Prepare result dictionary
        result = {
            "total_seconds": total_seconds,
            "total_hours": total_hours,
            "epoch_hours": epoch_hours,
            "formatted_total_time": self._format_time(total_seconds),
            "formatted_epoch_time": self._format_time(epoch_hours * 3600),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "time_per_sample": estimated_time_per_sample,
            "model_complexity_gflops": model_complexity / 1e9,  # Convert to GFLOPS
            "estimated_tflops": self.estimated_tflops,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory,
            "batch_size": self.batch_size,
            "dataset_size": self.dataset_size,
            "device": "gpu",
            "params_millions": params_millions
        }
        
        # Special considerations for A800 and H800 GPUs
        if "A800" in self.gpu_name and self.gpu_count > 1:
            result["note"] = "检测到A800 GPU，多卡训练时带宽限制可能影响性能，实际训练可能慢于估计值。"
        elif "H800" in self.gpu_name and self.gpu_count > 1:
            result["note"] = "检测到H800 GPU，多卡训练时带宽限制可能影响性能，实际训练可能慢于估计值。"
        
        # Add warning for low VRAM situations
        if hasattr(self.model_config, 'd_model') and self.model_config.d_model > 1024 and self.gpu_memory < 12:
            result["warning"] = f"模型可能过大，超出可用GPU显存 ({self.gpu_memory:.1f}GB)。考虑减小模型或批次大小。"
        
        return result
    
    def _calculate_model_complexity(self) -> float:
        """
        Calculate model complexity in terms of floating-point operations (FLOPs) per forward pass.
        
        Returns:
            float: Estimated FLOPs for a single forward pass
        """
        # Extract model configuration attributes safely
        d_model = getattr(self.model_config, 'd_model', 768)
        num_heads = getattr(self.model_config, 'num_heads', 12)
        d_ff = getattr(self.model_config, 'd_ff', d_model * 4)
        vocab_size = getattr(self.model_config, 'vocab_size', 50257)
        num_encoder_layers = getattr(self.model_config, 'num_encoder_layers', 6)
        num_decoder_layers = getattr(self.model_config, 'num_decoder_layers', 6)
        seq_len = getattr(self.model_config, 'max_seq_len', 512)
        
        # FLOPs for self-attention in each encoder layer
        encoder_self_attn_flops = 4 * seq_len * seq_len * d_model * num_encoder_layers
        
        # FLOPs for feed-forward networks in each encoder layer
        encoder_ffn_flops = 2 * seq_len * d_model * d_ff * num_encoder_layers
        
        # FLOPs for self-attention in each decoder layer
        decoder_self_attn_flops = 4 * seq_len * seq_len * d_model * num_decoder_layers
        
        # FLOPs for cross-attention in each decoder layer
        decoder_cross_attn_flops = 4 * seq_len * seq_len * d_model * num_decoder_layers
        
        # FLOPs for feed-forward networks in each decoder layer
        decoder_ffn_flops = 2 * seq_len * d_model * d_ff * num_decoder_layers
        
        # FLOPs for embeddings and output projections
        embedding_flops = seq_len * d_model * vocab_size
        output_projection_flops = seq_len * d_model * vocab_size
        
        # Total FLOPs for forward pass
        total_flops = (
            encoder_self_attn_flops + encoder_ffn_flops +
            decoder_self_attn_flops + decoder_cross_attn_flops + decoder_ffn_flops +
            embedding_flops + output_projection_flops
        )
        
        # Double for backward pass (approximation)
        total_flops_with_backward = total_flops * 2.5
        
        return total_flops_with_backward
    
    def _calculate_compute_per_sample(self, model_complexity: float) -> float:
        """
        Estimate compute time per sample based on model complexity.
        
        Args:
            model_complexity (float): Model complexity in FLOPs
            
        Returns:
            float: Estimated time per sample in seconds
        """
        # Base time per FLOP on a reference GPU (e.g., 10 TFLOPS)
        # Adjusted based on empirical measurements
        base_flop_time = 1e-13  # seconds per FLOP on a 10 TFLOPS GPU
        
        # Adjust for model size non-linearity
        # Larger models have more memory access overhead
        size_overhead = 1.0 + 0.1 * math.log2(max(1, model_complexity / 1e9))
        
        # Combine factors
        time_per_sample = model_complexity * base_flop_time * size_overhead
        
        return time_per_sample
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to a human-readable string.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
        else:
            days = seconds / 86400
            return f"{days:.1f}天"
    
    def _calculate_model_parameters(self) -> int:
        """
        Calculate the number of parameters in the model.
    
        Returns:
            int: Total number of parameters
        """
        # Extract model configuration attributes safely with defaults
        d_model = getattr(self.model_config, 'd_model', 768)
        num_heads = getattr(self.model_config, 'num_heads', 12)
        d_ff = getattr(self.model_config, 'd_ff', d_model * 4)
        vocab_size = getattr(self.model_config, 'vocab_size', 50257)
        num_encoder_layers = getattr(self.model_config, 'num_encoder_layers', 6)
        num_decoder_layers = getattr(self.model_config, 'num_decoder_layers', 6)
    
    # 其余计算代码保持不变...        
        # Embedding layer parameters
        embedding_params = d_model * vocab_size
        
        # Encoder self-attention parameters per layer
        encoder_self_attn_params = 4 * d_model * d_model  # Q, K, V projections and output projection
        
        # Encoder feed-forward parameters per layer
        encoder_ffn_params = 2 * d_model * d_ff  # Two linear transformations
        
        # Encoder layer norm parameters
        encoder_ln_params = 4 * d_model  # 2 layer norms per encoder layer, each with scale and bias
        
        # Total encoder parameters
        encoder_params = num_encoder_layers * (encoder_self_attn_params + encoder_ffn_params + encoder_ln_params)
        
        # Decoder self-attention parameters per layer
        decoder_self_attn_params = 4 * d_model * d_model
        
        # Decoder cross-attention parameters per layer
        decoder_cross_attn_params = 4 * d_model * d_model
        
        # Decoder feed-forward parameters per layer
        decoder_ffn_params = 2 * d_model * d_ff
        
        # Decoder layer norm parameters
        decoder_ln_params = 6 * d_model  # 3 layer norms per decoder layer
        
        # Total decoder parameters
        decoder_params = num_decoder_layers * (decoder_self_attn_params + decoder_cross_attn_params + decoder_ffn_params + decoder_ln_params)
        
        # Output projection parameters
        output_params = d_model * vocab_size
        
        # Total parameters
        total_params = embedding_params + encoder_params + decoder_params + output_params
        
        return total_params
    
    def get_resource_suggestion(self) -> Dict[str, Any]:
        """
        Suggest optimal batch size and other training parameters based on available resources.
        
        Returns:
            dict: Resource suggestions and optimization tips
        """
        # Calculate model memory requirements
        model_size_params = self._calculate_model_parameters()
        model_memory_gb = (model_size_params * 4 * 4) / (1024**3)  # 4 bytes per parameter, ~4x for optimizer states, gradients, etc.
        
        # Calculate batch memory requirements
        seq_len = getattr(self.model_config, 'max_seq_len', 512)
        d_model = getattr(self.model_config, 'd_model', 768)
        batch_element_size = seq_len * d_model * 4  # 4 bytes per float32
        batch_memory_gb = (batch_element_size * self.batch_size) / (1024**3)
        
        # Total memory needed for training
        total_memory_needed = model_memory_gb + batch_memory_gb
        
        # Make suggestions
        suggestions = {
            "model_parameters": model_size_params,
            "model_size_gb": model_memory_gb,
            "batch_memory_gb": batch_memory_gb,
            "total_memory_needed_gb": total_memory_needed,
            "available_gpu_memory_gb": self.gpu_memory,
            "tips": []
        }
        
        # Check if model fits in memory
        if self.gpu_available:
            if total_memory_needed > self.gpu_memory * 0.9:
                suggestions["tips"].append(f"警告: 所需内存 ({total_memory_needed:.2f}GB) 接近或超过可用GPU显存 ({self.gpu_memory:.2f}GB)")
                
                # Suggest smaller batch size
                max_batch_size = max(1, int((self.gpu_memory * 0.8 - model_memory_gb) / (batch_element_size / (1024**3))))
                if max_batch_size < self.batch_size:
                    suggestions["tips"].append(f"建议: 将批次大小从 {self.batch_size} 减小到 {max_batch_size}")
                    suggestions["recommended_batch_size"] = max_batch_size
                
                # Suggest smaller model if still necessary
                if max_batch_size <= 1:
                    smaller_d_model = int(d_model * 0.75)
                    smaller_layers = max(1, int(min(getattr(self.model_config, 'num_encoder_layers', 6), 
                                                   getattr(self.model_config, 'num_decoder_layers', 6)) * 0.75))
                    suggestions["tips"].append(f"建议: 减小模型大小 (d_model: {d_model} -> {smaller_d_model}, layers: {getattr(self.model_config, 'num_encoder_layers', 6)}/{getattr(self.model_config, 'num_decoder_layers', 6)} -> {smaller_layers})")
                    suggestions["recommended_d_model"] = smaller_d_model
                    suggestions["recommended_layers"] = smaller_layers
            else:
                # Suggest optimal batch size for better utilization
                optimal_batch_size = min(
                    2048,  # Upper limit for diminishing returns
                    max(4, int((self.gpu_memory * 0.8 - model_memory_gb) / (batch_element_size / (1024**3))))
                )
                if optimal_batch_size > self.batch_size * 1.5:  # Only suggest if significantly better
                    suggestions["tips"].append(f"提示: 将批次大小从 {self.batch_size} 增加到 {optimal_batch_size} 可能提高训练效率")
                    suggestions["recommended_batch_size"] = optimal_batch_size
        else:
            suggestions["tips"].append("提示: 使用GPU训练将显著提高训练速度")
        
        # Suggest mixed precision training for newer GPUs
        if self.gpu_available:
            # Special advice for A800 and H800 GPUs
            if "A800" in self.gpu_name:
                suggestions["tips"].append("提示: 检测到A800 GPU (A100的中国特供版)，单卡训练性能接近A100")
                if self.gpu_count > 1:
                    suggestions["tips"].append("注意: A800的NVLink带宽限制为400GB/s (vs A100的600GB/s)，多卡训练可能受影响")
                suggestions["tips"].append("建议: 启用BF16混合精度训练以提高性能")
                suggestions["recommend_mixed_precision"] = "bfloat16"
            elif "H800" in self.gpu_name:
                suggestions["tips"].append("提示: 检测到H800 GPU (H100的中国特供版)，单卡训练性能接近H100")
                if self.gpu_count > 1:
                    suggestions["tips"].append("注意: H800的NVLink带宽受限，多卡训练可能受影响")
                suggestions["tips"].append("建议: 启用FP8精度训练以充分发挥H800性能")
                suggestions["recommend_mixed_precision"] = "float8"
            elif any(gpu in self.gpu_name for gpu in ["RTX", "A100", "H100", "V100", "MI"]):
                suggestions["tips"].append("提示: 启用混合精度训练 (float16) 可减少内存使用并加快训练速度")
                suggestions["recommend_mixed_precision"] = "float16"
        
        # Suggest gradient accumulation for large models/small batches
        if self.batch_size < 8 and total_memory_needed > self.gpu_memory * 0.7:
            suggestions["tips"].append("提示: 使用梯度累积可在较小批次大小下模拟更大批次训练")
            suggestions["recommend_gradient_accumulation"] = 4  # Reasonable default
        
        return suggestions
    
    def run_benchmark(self, num_iterations: int = 10, print_results: bool = True) -> Dict[str, float]:
        """
        Run a quick benchmark to calibrate estimation accuracy.
        
        Args:
            num_iterations (int): Number of benchmark iterations
            print_results (bool): Whether to print benchmark results
            
        Returns:
            dict: Benchmark results
        """
        if not self.gpu_available:
            if print_results:
                print("Benchmark requires GPU. Skipping.")
            return {"error": "No GPU available for benchmarking"}
        
        try:
            # Create small model for benchmarking
            import torch
            import torch.nn as nn
            
            class BenchmarkModel(nn.Module):
                def __init__(self, d_model, n_layers):
                    super().__init__()
                    self.layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=8,
                            dim_feedforward=d_model*4,
                            batch_first=True
                        ) for _ in range(n_layers)
                    ])
                
                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    return x
            
            # Create a model similar to APT but smaller
            d_model = min(getattr(self.model_config, 'd_model', 768), 256)  # Limit size for benchmark
            n_layers = min(getattr(self.model_config, 'num_encoder_layers', 6), 2)
            model = BenchmarkModel(d_model, n_layers).to('cuda')
            
            # Create dummy input
            seq_len = min(getattr(self.model_config, 'max_seq_len', 512), 128)
            batch_size = min(self.batch_size, 8)
            dummy_input = torch.randn(batch_size, seq_len, d_model).to('cuda')
            
            # Warmup
            for _ in range(5):
                output = model(dummy_input)
                
            # Benchmark forward pass
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(num_iterations):
                output = model(dummy_input)
                torch.cuda.synchronize()
            forward_time = (time.time() - start_time) / num_iterations
            
            # Benchmark forward+backward pass
            dummy_target = torch.randn_like(dummy_input)
            loss_fn = nn.MSELoss()
            
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(num_iterations):
                output = model(dummy_input)
                loss = loss_fn(output, dummy_target)
                loss.backward()
                torch.cuda.synchronize()
            total_time = (time.time() - start_time) / num_iterations
            
            # Calculate TFLOPS based on benchmark
            # Approximate FLOPs for one transformer layer
            flops_per_layer = 4 * 2 * batch_size * seq_len * seq_len * d_model
            flops_per_layer += 2 * 2 * batch_size * seq_len * d_model * (d_model * 4)
            total_flops = flops_per_layer * n_layers
            measured_tflops = total_flops / (total_time * 1e12)
            
            results = {
                "forward_time": forward_time,
                "total_time": total_time,
                "backward_time": total_time - forward_time,
                "backward_forward_ratio": (total_time - forward_time) / forward_time,
                "measured_tflops": measured_tflops,
                "estimated_tflops": self.estimated_tflops,
                "tflops_accuracy": measured_tflops / self.estimated_tflops if self.estimated_tflops > 0 else 0,
            }
            
            if print_results:
                print("\n" + "="*50)
                print("GPU性能基准测试")
                print("="*50)
                print(f"模型: {n_layers} 层, d_model={d_model}")
                print(f"输入: 批次大小={batch_size}, 序列长度={seq_len}")
                print(f"前向传播时间: {forward_time*1000:.2f}ms")
                print(f"反向传播时间: {(total_time-forward_time)*1000:.2f}ms")
                print(f"总时间: {total_time*1000:.2f}ms")
                print(f"测量性能: {measured_tflops:.2f} TFLOPS")
                print(f"估计性能: {self.estimated_tflops:.2f} TFLOPS")
                print(f"估计准确率: {results['tflops_accuracy']*100:.1f}%")
                
                # Update estimates if accurate measurement is available
                if 0.5 < results['tflops_accuracy'] < 2.0:
                    self.estimated_tflops = measured_tflops
                    print(f"已更新性能估计为: {measured_tflops:.2f} TFLOPS")
                
                print("="*50)
            
            return results
            
        except Exception as e:
            if print_results:
                print(f"基准测试失败: {str(e)}")
            return {"error": str(e)}
    
    def _save_calibration(self, results):
        """
        Save calibration results to file for future use.
        
        Args:
            results: Benchmark results
        """
        import json
        import os
        
        # Create cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".apt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing calibration data
        calibration_path = os.path.join(cache_dir, "calibration.json")
        calibration = {}
        
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, 'r') as f:
                    calibration = json.load(f)
            except Exception:
                pass
        
        # Update calibration data
        gpu_key = self.gpu_name.replace(" ", "_")
        calibration[gpu_key] = {
            "gpu_name": self.gpu_name,
            "estimated_tflops": self.estimated_tflops,
            "measured_tflops": results.get("measured_tflops", 0),
            "timestamp": time.time()
        }
        
        # Save calibration data
        with open(calibration_path, 'w') as f:
            json.dump(calibration, f, indent=2)
    
    def print_estimation(self) -> Dict[str, Any]:
        """
        Print a detailed training time estimation report.
        
        Returns:
            dict: The estimation results
        """
        estimation = self.estimate_training_time()
        resource_suggestion = self.get_resource_suggestion()
        
        print("\n" + "="*50)
        print("训练时间预估")
        print("="*50)
        
        print(f"\n模型信息:")
        print(f"  参数数量: {resource_suggestion['model_parameters']:,}")
        print(f"  模型大小: {resource_suggestion['model_size_gb']:.2f}GB")
        print(f"  隐藏维度: {getattr(self.model_config, 'd_model', 768)}")
        print(f"  编码器层数: {getattr(self.model_config, 'num_encoder_layers', 6)}")
        print(f"  解码器层数: {getattr(self.model_config, 'num_decoder_layers', 6)}")
        print(f"  总计算量: {estimation.get('model_complexity_gflops', 0):.2f} GFLOPS")
        
        print(f"\n数据集信息:")
        print(f"  样本数: {self.dataset_size:,}")
        print(f"  批次大小: {self.batch_size}")
        print(f"  训练轮数: {self.epochs}")
        print(f"  每轮步数: {estimation['steps_per_epoch']:.0f}")
        print(f"  总步数: {estimation['total_steps']:.0f}")
        print(f"  批次内存占用: {resource_suggestion['batch_memory_gb']:.2f}GB")
        
        print(f"\n硬件信息:")
        if self.gpu_available:
            print(f"  GPU: {self.gpu_name} x{self.gpu_count}")
            print(f"  显存: {self.gpu_memory:.1f}GB")
            print(f"  估计性能: {self.estimated_tflops:.1f} TFLOPS")
            
            # 显示A800/H800特殊信息
            if "A800" in self.gpu_name:
                print(f"  注意: A800是A100的中国特供版，单GPU性能相同，但多GPU通信带宽受限")
            elif "H800" in self.gpu_name:
                print(f"  注意: H800是H100的中国特供版，建议启用FP8精度训练以获得最佳性能")
        else:
            print(f"  使用CPU训练 ({self.cpu_cores} 核心)")
            print(f"  系统内存: {self.ram_gb:.1f}GB")
        
        print(f"\n预估训练时间:")
        print(f"  总时间: {estimation['formatted_total_time']}")
        print(f"  每轮时间: {estimation['formatted_epoch_time']}")
        
        print(f"\n资源建议:")
        for tip in resource_suggestion['tips']:
            print(f"  • {tip}")
        
        if 'warning' in estimation:
            print(f"\n警告: {estimation['warning']}")
        
        if not self.gpu_available:
            print(f"\n警告: 使用CPU训练，实际时间可能远超估计值")
        elif self.gpu_memory < 8 and resource_suggestion['model_size_gb'] > 2:
            print(f"\n警告: GPU显存较小 ({self.gpu_memory:.1f}GB)，可能影响训练速度或导致内存不足")
        
        print("="*50)
        
        return estimation