"""
Model Distillation Plugin for APT
知识蒸馏插件 - 将大模型的知识转移到小模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader


class ModelDistillationPlugin:
    """
    模型蒸馏插件
    
    支持三种蒸馏方法:
    1. 响应蒸馏 (Response Distillation) - 蒸馏输出logits
    2. 特征蒸馏 (Feature Distillation) - 蒸馏中间层特征
    3. 关系蒸馏 (Relation Distillation) - 蒸馏样本间的关系
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "model-distillation"
        self.version = "1.0.0"
        self.config = config
        
        # 蒸馏参数
        self.temperature = config.get('temperature', 4.0)
        self.alpha = config.get('alpha', 0.5)  # 蒸馏损失权重
        self.beta = config.get('beta', 0.5)   # 真实标签损失权重
        
        self.distill_type = config.get('distill_type', 'response')  # response/feature/relation
    
    # ==================== 响应蒸馏 ====================
    
    def response_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        响应蒸馏损失 (最常用)
        
        使用KL散度衡量学生和教师输出分布的差异
        
        Args:
            student_logits: 学生模型的logits [batch, seq_len, vocab]
            teacher_logits: 教师模型的logits [batch, seq_len, vocab]
            labels: 真实标签 (可选)
            temperature: 温度参数,用于软化概率分布
            
        Returns:
            蒸馏损失
        """
        T = temperature or self.temperature
        
        # 使用温度参数软化分布
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        
        # KL散度损失 (衡量两个分布的差异)
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (T ** 2)  # 温度的平方用于缩放
        
        # 如果有真实标签,结合交叉熵损失
        if labels is not None:
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            # 加权组合
            total_loss = self.alpha * distill_loss + self.beta * ce_loss
            return total_loss
        
        return distill_loss
    
    # ==================== 特征蒸馏 ====================
    
    def feature_distillation_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        特征蒸馏损失
        
        让学生模型的中间层特征接近教师模型
        
        Args:
            student_features: 学生模型特征 [batch, seq_len, hidden_dim]
            teacher_features: 教师模型特征 [batch, seq_len, hidden_dim]
            normalize: 是否归一化特征
            
        Returns:
            特征蒸馏损失
        """
        if normalize:
            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        
        # MSE损失
        feature_loss = F.mse_loss(student_features, teacher_features)
        
        return feature_loss
    
    def multi_layer_feature_distillation(
        self,
        student_features_list: list,
        teacher_features_list: list,
        layer_weights: Optional[list] = None
    ) -> torch.Tensor:
        """
        多层特征蒸馏
        
        Args:
            student_features_list: 学生模型多层特征列表
            teacher_features_list: 教师模型多层特征列表
            layer_weights: 各层权重
            
        Returns:
            总特征损失
        """
        if layer_weights is None:
            layer_weights = [1.0] * len(student_features_list)
        
        total_loss = 0
        for s_feat, t_feat, weight in zip(
            student_features_list, teacher_features_list, layer_weights
        ):
            layer_loss = self.feature_distillation_loss(s_feat, t_feat)
            total_loss += weight * layer_loss
        
        return total_loss / len(student_features_list)
    
    # ==================== 关系蒸馏 ====================
    
    def relation_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        关系蒸馏损失
        
        保持样本间的相对关系(相似度)
        
        Args:
            student_outputs: 学生模型输出 [batch, hidden]
            teacher_outputs: 教师模型输出 [batch, hidden]
            
        Returns:
            关系蒸馏损失
        """
        # 计算样本间的相似度矩阵
        student_sim = self._compute_similarity_matrix(student_outputs)
        teacher_sim = self._compute_similarity_matrix(teacher_outputs)
        
        # L2损失
        relation_loss = F.mse_loss(student_sim, teacher_sim)
        
        return relation_loss
    
    def _compute_similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """计算特征的相似度矩阵"""
        # 归一化
        features = F.normalize(features, p=2, dim=-1)
        
        # 计算余弦相似度矩阵
        similarity = torch.matmul(features, features.transpose(-2, -1))
        
        return similarity
    
    # ==================== 注意力蒸馏 ====================
    
    def attention_distillation_loss(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """
        注意力蒸馏损失
        
        让学生模型学习教师模型的注意力模式
        
        Args:
            student_attention: 学生注意力权重 [batch, heads, seq, seq]
            teacher_attention: 教师注意力权重 [batch, heads, seq, seq]
            
        Returns:
            注意力蒸馏损失
        """
        # MSE损失
        attention_loss = F.mse_loss(student_attention, teacher_attention)
        
        return attention_loss
    
    # ==================== 蒸馏训练流程 ====================
    
    def distill_training_step(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        单步蒸馏训练
        
        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            batch: 训练批次数据
            optimizer: 优化器
            
        Returns:
            损失字典
        """
        student_model.train()
        teacher_model.eval()
        
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)
        
        # 教师模型前向传播 (不计算梯度)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, output_hidden_states=True)
            teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs[0]
            teacher_features = teacher_outputs.hidden_states if hasattr(teacher_outputs, 'hidden_states') else None
        
        # 学生模型前向传播
        student_outputs = student_model(input_ids, output_hidden_states=True)
        student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs[0]
        student_features = student_outputs.hidden_states if hasattr(student_outputs, 'hidden_states') else None
        
        # 计算蒸馏损失
        losses = {}
        
        if self.distill_type == 'response':
            # 响应蒸馏
            loss = self.response_distillation_loss(
                student_logits, teacher_logits, labels
            )
            losses['response_loss'] = loss.item()
        
        elif self.distill_type == 'feature':
            # 特征蒸馏
            if student_features and teacher_features:
                loss = self.multi_layer_feature_distillation(
                    student_features, teacher_features
                )
                losses['feature_loss'] = loss.item()
            else:
                raise ValueError("模型需要支持输出hidden_states")
        
        elif self.distill_type == 'combined':
            # 组合蒸馏
            response_loss = self.response_distillation_loss(
                student_logits, teacher_logits, labels
            )
            
            feature_loss = torch.tensor(0.0, device=input_ids.device)
            if student_features and teacher_features:
                feature_loss = self.multi_layer_feature_distillation(
                    student_features, teacher_features
                )
            
            loss = response_loss + 0.1 * feature_loss
            losses['response_loss'] = response_loss.item()
            losses['feature_loss'] = feature_loss.item()
        
        else:
            raise ValueError(f"未知的蒸馏类型: {self.distill_type}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses['total_loss'] = loss.item()
        return losses
    
    def distill_model(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 3,
        device: str = 'cuda'
    ):
        """
        完整的模型蒸馏流程
        
        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            train_dataloader: 训练数据加载器
            optimizer: 优化器
            num_epochs: 训练轮数
            device: 设备
        """
        print("🎓 开始知识蒸馏训练...")
        
        student_model.to(device)
        teacher_model.to(device)
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                # 将数据移到设备
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 训练步骤
                losses = self.distill_training_step(
                    student_model, teacher_model, batch, optimizer
                )
                epoch_losses.append(losses['total_loss'])
                
                # 日志
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(train_dataloader)} | "
                          f"Loss: {losses['total_loss']:.4f}")
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"📊 Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        print("✅ 知识蒸馏完成!")
    
    # ==================== 工具方法 ====================
    
    def compress_model(
        self,
        teacher_model: nn.Module,
        compression_ratio: float = 0.5
    ) -> nn.Module:
        """
        根据压缩比创建学生模型
        
        Args:
            teacher_model: 教师模型
            compression_ratio: 压缩比 (0-1)
            
        Returns:
            学生模型
        """
        # TODO: 根据教师模型架构自动创建压缩后的学生模型
        # 这需要了解具体的模型结构
        pass
    
    # ==================== 插件钩子 ====================
    
    def on_training_start(self, context: Dict[str, Any]):
        """训练开始时初始化蒸馏"""
        if context.get('enable_distillation'):
            print("🎓 启用知识蒸馏模式")
            self.teacher_model = context.get('teacher_model')
            if self.teacher_model:
                self.teacher_model.eval()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置
    config = {
        'temperature': 4.0,
        'alpha': 0.7,      # 蒸馏损失权重
        'beta': 0.3,       # 真实标签权重
        'distill_type': 'response',  # response/feature/combined
    }
    
    plugin = ModelDistillationPlugin(config)
    
    # 示例:响应蒸馏
    batch_size, seq_len, vocab_size = 8, 128, 50000
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = plugin.response_distillation_loss(
        student_logits, teacher_logits, labels
    )
    print(f"响应蒸馏损失: {loss.item():.4f}")
    
    print("\n✅ 知识蒸馏插件示例完成!")
    print("\n💡 使用建议:")
    print("1. 响应蒸馏适合大多数场景,效果好且简单")
    print("2. 特征蒸馏适合模型结构相似的情况")
    print("3. 组合蒸馏效果最好,但训练较慢")
    print("4. 温度参数T建议设置为2-8之间")
