"""
Model Pruning Plugin for APT
模型剪枝插件 - 减少模型参数量,提升推理速度
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, List, Tuple
import numpy as np


class ModelPruningPlugin:
    """
    模型剪枝插件
    
    支持的剪枝策略:
    1. 结构化剪枝 (Structured Pruning) - 剪除整个通道/层
    2. 非结构化剪枝 (Unstructured Pruning) - 剪除单个权重
    3. 权重大小剪枝 (Magnitude-based) - 基于权重绝对值
    4. Taylor剪枝 (Taylor Pruning) - 基于一阶泰勒展开
    5. 彩票假说剪枝 (Lottery Ticket Hypothesis)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "model-pruning"
        self.version = "1.0.0"
        self.config = config
        
        self.prune_ratio = config.get('prune_ratio', 0.3)  # 剪枝比例
        self.prune_type = config.get('prune_type', 'magnitude')  # 剪枝类型
        self.structured = config.get('structured', False)  # 是否结构化剪枝
    
    # ==================== 权重大小剪枝 ====================
    
    def magnitude_pruning(
        self,
        model: nn.Module,
        prune_ratio: float,
        structured: bool = False
    ) -> nn.Module:
        """
        基于权重大小的剪枝
        
        移除绝对值最小的权重
        
        Args:
            model: 待剪枝模型
            prune_ratio: 剪枝比例 (0-1)
            structured: 是否结构化剪枝
            
        Returns:
            剪枝后的模型
        """
        print(f"✂️ 开始权重大小剪枝 (剪枝比例: {prune_ratio*100:.1f}%)")
        
        parameters_to_prune = []
        
        # 收集所有需要剪枝的参数
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        if structured:
            # 结构化剪枝 (剪除整个输出通道)
            for module, param_name in parameters_to_prune:
                prune.ln_structured(
                    module, name=param_name,
                    amount=prune_ratio, n=2, dim=0
                )
        else:
            # 非结构化剪枝 (L1范数)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_ratio,
            )
        
        # 计算实际剪枝比例
        pruned_params, total_params = self._count_pruned_params(model)
        actual_ratio = pruned_params / total_params
        
        print(f"✅ 剪枝完成! 剪除参数: {pruned_params}/{total_params} ({actual_ratio*100:.2f}%)")
        
        return model
    
    # ==================== Taylor剪枝 ====================
    
    def taylor_pruning(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        prune_ratio: float,
        device: str = 'cuda'
    ) -> nn.Module:
        """
        基于一阶泰勒展开的剪枝
        
        使用梯度信息评估参数重要性:
        importance = |weight * gradient|
        
        Args:
            model: 待剪枝模型
            dataloader: 数据加载器(用于计算梯度)
            prune_ratio: 剪枝比例
            device: 设备
            
        Returns:
            剪枝后的模型
        """
        print(f"✂️ 开始Taylor剪枝 (剪枝比例: {prune_ratio*100:.1f}%)")
        
        model.to(device)
        model.train()
        
        # 1. 计算参数的Taylor重要性分数
        importance_scores = {}
        
        # 前向传播并计算梯度
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # 使用10个批次估算重要性
                break
            
            # 准备输入
            if isinstance(batch, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                outputs = model(**inputs)
            else:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                outputs = model(inputs)
            
            # 计算损失
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            loss.backward()
        
        # 2. 计算每个参数的重要性: |weight * gradient|
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                importance = torch.abs(param * param.grad)
                importance_scores[name] = importance.detach().cpu()
        
        # 清除梯度
        model.zero_grad()
        
        # 3. 全局排序并剪枝
        all_scores = torch.cat([scores.flatten() for scores in importance_scores.values()])
        threshold = torch.quantile(all_scores, prune_ratio)
        
        # 4. 应用剪枝
        for name, param in model.named_parameters():
            if name in importance_scores:
                mask = (importance_scores[name] > threshold).float().to(device)
                param.data *= mask
        
        print("✅ Taylor剪枝完成!")
        
        return model
    
    # ==================== 结构化剪枝 ====================
    
    def structured_channel_pruning(
        self,
        model: nn.Module,
        prune_ratio: float,
        criterion: str = 'l1'
    ) -> nn.Module:
        """
        结构化通道剪枝
        
        剪除整个卷积/线性层的输出通道
        
        Args:
            model: 待剪枝模型
            prune_ratio: 剪枝比例
            criterion: 剪枝标准 ('l1', 'l2', 'random')
            
        Returns:
            剪枝后的模型
        """
        print(f"✂️ 开始结构化通道剪枝 (标准: {criterion})")
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # 计算每个输出通道的范数
                weight = module.weight.data
                out_channels = weight.shape[0]
                
                if criterion == 'l1':
                    # L1范数
                    channel_norms = torch.norm(weight.view(out_channels, -1), p=1, dim=1)
                elif criterion == 'l2':
                    # L2范数
                    channel_norms = torch.norm(weight.view(out_channels, -1), p=2, dim=1)
                elif criterion == 'random':
                    # 随机
                    channel_norms = torch.rand(out_channels)
                
                # 选择要保留的通道
                num_keep = int(out_channels * (1 - prune_ratio))
                _, keep_indices = torch.topk(channel_norms, num_keep)
                keep_indices = sorted(keep_indices.tolist())
                
                # 剪枝 (实际实现中需要调整后续层的输入维度)
                # 这里只是示意,真实情况需要更复杂的处理
                prune.ln_structured(
                    module, name='weight',
                    amount=prune_ratio, n=2, dim=0
                )
        
        print("✅ 结构化剪枝完成!")
        
        return model
    
    # ==================== 彩票假说剪枝 ====================
    
    def lottery_ticket_pruning(
        self,
        model: nn.Module,
        train_function: callable,
        prune_iterations: int = 5,
        prune_ratio_per_iter: float = 0.2
    ) -> Tuple[nn.Module, Dict]:
        """
        彩票假说剪枝 (Lottery Ticket Hypothesis)
        
        迭代式剪枝:
        1. 训练模型到收敛
        2. 剪枝一部分权重
        3. 将剩余权重重置到初始值
        4. 重复步骤1-3
        
        Args:
            model: 待剪枝模型
            train_function: 训练函数
            prune_iterations: 迭代次数
            prune_ratio_per_iter: 每次迭代的剪枝比例
            
        Returns:
            (剪枝后的模型, 历史记录)
        """
        print("🎰 开始彩票假说剪枝...")
        
        # 保存初始权重
        initial_weights = {name: param.data.clone() 
                          for name, param in model.named_parameters()}
        
        history = {
            'iteration': [],
            'total_params': [],
            'remaining_params': [],
            'accuracy': []
        }
        
        for iteration in range(prune_iterations):
            print(f"\n🎯 迭代 {iteration+1}/{prune_iterations}")
            
            # 1. 训练模型
            print("  训练中...")
            accuracy = train_function(model)
            
            # 2. 剪枝
            print(f"  剪枝 {prune_ratio_per_iter*100:.1f}%...")
            model = self.magnitude_pruning(model, prune_ratio_per_iter)
            
            # 3. 重置未剪枝的权重到初始值
            print("  重置权重到初始值...")
            for name, param in model.named_parameters():
                if name in initial_weights:
                    # 获取剪枝掩码
                    if hasattr(param, '_forward_pre_hooks'):
                        # 只重置未被剪枝的权重
                        mask = torch.ones_like(param.data)
                        for hook in param._forward_pre_hooks.values():
                            if hasattr(hook, 'mask'):
                                mask = hook.mask
                        param.data = initial_weights[name] * mask
            
            # 记录统计
            total_params, remaining_params = self._count_total_params(model)
            history['iteration'].append(iteration + 1)
            history['total_params'].append(total_params)
            history['remaining_params'].append(remaining_params)
            history['accuracy'].append(accuracy)
            
            print(f"  剩余参数: {remaining_params}/{total_params} "
                  f"({remaining_params/total_params*100:.2f}%)")
        
        print("\n✅ 彩票假说剪枝完成!")
        
        return model, history
    
    # ==================== 剪枝后处理 ====================
    
    def make_pruning_permanent(self, model: nn.Module) -> nn.Module:
        """
        永久应用剪枝 (移除掩码,真正删除参数)
        
        Args:
            model: 剪枝后的模型
            
        Returns:
            永久剪枝的模型
        """
        print("🔧 永久应用剪枝...")
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                try:
                    # 移除剪枝的重参数化
                    prune.remove(module, 'weight')
                except:
                    pass
        
        print("✅ 剪枝已永久应用!")
        
        return model
    
    def fine_tune_after_pruning(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 3,
        device: str = 'cuda'
    ) -> nn.Module:
        """
        剪枝后微调
        
        剪枝会损失一些精度,需要微调恢复
        
        Args:
            model: 剪枝后的模型
            train_dataloader: 训练数据
            optimizer: 优化器
            num_epochs: 微调轮数
            device: 设备
            
        Returns:
            微调后的模型
        """
        print(f"🎯 开始剪枝后微调 ({num_epochs} epochs)...")
        
        model.to(device)
        model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # 准备数据
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    outputs = model(**batch)
                else:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                
                # 计算损失
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(train_dataloader)} | "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"📊 Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        print("✅ 微调完成!")
        
        return model
    
    # ==================== 工具方法 ====================
    
    def _count_pruned_params(self, model: nn.Module) -> Tuple[int, int]:
        """计算被剪枝的参数数量"""
        pruned = 0
        total = 0
        
        for module in model.modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                if hasattr(weight, '_forward_pre_hooks'):
                    # 有剪枝掩码
                    for hook in weight._forward_pre_hooks.values():
                        if hasattr(hook, 'mask'):
                            mask = hook.mask
                            pruned += (mask == 0).sum().item()
                            total += mask.numel()
                else:
                    # 检查是否为0
                    pruned += (weight == 0).sum().item()
                    total += weight.numel()
        
        return pruned, total
    
    def _count_total_params(self, model: nn.Module) -> Tuple[int, int]:
        """计算总参数和剩余参数"""
        total = sum(p.numel() for p in model.parameters())
        pruned, _ = self._count_pruned_params(model)
        remaining = total - pruned
        return total, remaining
    
    def get_pruning_statistics(self, model: nn.Module) -> Dict:
        """获取剪枝统计信息"""
        stats = {
            'layer_stats': [],
            'total_params': 0,
            'pruned_params': 0,
            'remaining_params': 0,
            'sparsity': 0.0
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                layer_pruned = (weight == 0).sum().item()
                layer_total = weight.numel()
                
                stats['layer_stats'].append({
                    'name': name,
                    'total': layer_total,
                    'pruned': layer_pruned,
                    'sparsity': layer_pruned / layer_total if layer_total > 0 else 0
                })
                
                stats['total_params'] += layer_total
                stats['pruned_params'] += layer_pruned
        
        stats['remaining_params'] = stats['total_params'] - stats['pruned_params']
        stats['sparsity'] = stats['pruned_params'] / stats['total_params'] if stats['total_params'] > 0 else 0
        
        return stats
    
    # ==================== 插件钩子 ====================
    
    def on_training_end(self, context: Dict[str, Any]):
        """训练结束后自动剪枝"""
        if self.config.get('auto_prune', False):
            model = context.get('model')
            print("\n🚀 自动触发模型剪枝...")
            
            model = self.magnitude_pruning(
                model,
                prune_ratio=self.prune_ratio,
                structured=self.structured
            )
            
            # 保存剪枝后的模型
            save_path = context.get('output_dir', './pruned_model')
            torch.save(model.state_dict(), f"{save_path}/pruned_model.pt")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置
    config = {
        'prune_ratio': 0.3,  # 剪枝30%的参数
        'prune_type': 'magnitude',  # magnitude/taylor/structured/lottery
        'structured': False,  # 非结构化剪枝
        'auto_prune': True,  # 训练后自动剪枝
    }
    
    plugin = ModelPruningPlugin(config)
    
    # 创建测试模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    
    print("原始模型参数量:", sum(p.numel() for p in model.parameters()))
    
    # 应用剪枝
    model = plugin.magnitude_pruning(model, prune_ratio=0.3, structured=False)
    
    # 获取统计
    stats = plugin.get_pruning_statistics(model)
    print(f"\n剪枝统计:")
    print(f"  总参数: {stats['total_params']}")
    print(f"  剪枝参数: {stats['pruned_params']}")
    print(f"  剩余参数: {stats['remaining_params']}")
    print(f"  稀疏度: {stats['sparsity']*100:.2f}%")
    
    print("\n✅ 模型剪枝插件示例完成!")
