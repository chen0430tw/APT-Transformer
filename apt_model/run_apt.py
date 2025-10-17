import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 获取modeling目录路径并添加到Python路径
modeling_dir = os.path.join(current_dir, "modeling")
sys.path.insert(0, modeling_dir)

import torch
import torch.nn as nn

try:
    print("导入apt_model模块...")
    import apt_model
    print("✓ 成功导入apt_model模块")
    
    if hasattr(apt_model, 'APTModel'):
        print("✓ APTModel类存在")
        
        # 尝试创建模型
        print("\n尝试创建模型实例...")
        
        # 创建更完整的配置
        class CompleteConfig:
            def __init__(self):
                # 基本参数
                self.vocab_size = 30000
                self.d_model = 512
                self.num_encoder_layers = 2
                self.num_decoder_layers = 2
                self.num_heads = 8
                self.d_ff = 2048
                self.max_seq_len = 512
                self.dropout = 0.1
                
                # 自生成变换参数
                self.activation = "gelu"
                self.epsilon = 1e-6
                self.alpha = 0.1
                self.beta = 0.01
                self.init_tau = 1.0
                self.sr_ratio = 4
                self.use_autopoietic = True
                self.batch_first = True
                self.pad_token_id = 0
                self.bos_token_id = 101
                self.eos_token_id = 102
                self.base_lr = 3e-5
        
        try:
            print("✓ PyTorch版本:", torch.__version__)
            
            config = CompleteConfig()
            print("配置创建成功。尝试初始化模型...")
            model = apt_model.APTModel(config)
            print(f"✓ 模型创建成功! 参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 显示模型层次结构
            def print_model_structure(model, prefix=''):
                for name, child in model.named_children():
                    print(f"{prefix}├─ {name}")
                    if list(child.named_children()):
                        print_model_structure(child, prefix + '│  ')
            
            print("\n模型结构:")
            print_model_structure(model)
            
            # 测试简单的前向传播
            print("\n尝试前向传播...")
            batch_size = 2
            src_len = 10
            tgt_len = 8
            
            src_ids = torch.randint(0, config.vocab_size, (batch_size, src_len))
            tgt_ids = torch.randint(0, config.vocab_size, (batch_size, tgt_len))
            
            print(f"- 源序列形状: {src_ids.shape}")
            print(f"- 目标序列形状: {tgt_ids.shape}")
            
            with torch.no_grad():
                try:
                    outputs = model(src_ids, tgt_ids)
                    print(f"✓ 前向传播成功!")
                    print(f"- 输出logits形状: {outputs['logits'].shape}")
                    print(f"- 编码器输出形状: {outputs['encoder_output'].shape}")
                    print(f"- 解码器输出形状: {outputs['decoder_output'].shape}")
                    
                    # 可视化注意力权重的形状
                    print("\n尝试获取注意力权重...")
                    # 通过将need_weights设置为True来获取注意力权重
                    encoder_layer = model.encoder_layers[0]
                    dummy_src = torch.zeros(batch_size, src_len, config.d_model)
                    attn_output, attn_weights = encoder_layer.self_attn(
                        query=dummy_src,
                        key=dummy_src,
                        value=dummy_src,
                        need_weights=True
                    )
                    if attn_weights is not None:
                        print(f"✓ 注意力权重形状: {attn_weights.shape}")
                    else:
                        print("✗ 注意力权重为None")
                    
                except Exception as e:
                    print(f"✗ 前向传播失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
            # 测试自生成机制
            print("\n测试自生成变换机制...")
            try:
                # 找到模型中的自生成注意力层
                for name, module in model.named_modules():
                    if isinstance(module, apt_model.AutopoieticAttention):
                        print(f"发现自生成注意力层: {name}")
                        
                        # 创建随机注意力分数
                        rand_attn_scores = torch.rand(batch_size, config.num_heads, src_len, src_len)
                        print(f"- 随机注意力分数形状: {rand_attn_scores.shape}")
                        
                        # 应用自生成变换
                        transformed_scores = module.autopoietic_transform(rand_attn_scores)
                        print(f"✓ 自生成变换成功!")
                        print(f"- 变换后注意力分数形状: {transformed_scores.shape}")
                        
                        # 检查变换前后的差异
                        diff = (transformed_scores - rand_attn_scores).abs().mean().item()
                        print(f"- 变换前后平均差异: {diff:.6f}")
                        
                        # 只测试第一个找到的层
                        break
                else:
                    print("✗ 未找到任何自生成注意力层")
            
            except Exception as e:
                print(f"✗ 测试自生成变换失败: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"✗ 模型创建失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ APTModel类不存在")
        print(f"模块内容: {dir(apt_model)}")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")