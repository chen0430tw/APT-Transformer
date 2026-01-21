#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：MXFP4 + GPU优化MoE + 100K GPU支持

测试内容:
1. MXFP4 量化功能
2. GPU 优化版 MoE 层
3. 超大规模训练配置
4. Virtual Blackwell 集成

作者: chen0430tw
日期: 2026-01-21
"""

import torch
import torch.nn as nn
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_mxfp4_quantization():
    """测试 MXFP4 量化"""
    print("\n" + "=" * 70)
    print("测试 1: MXFP4 量化")
    print("=" * 70)

    try:
        from apt_model.optimization.mxfp4_quantization import (
            MXFP4Quantizer,
            MXFP4Config,
            MXFP4Linear
        )

        # 测试基本量化
        quantizer = MXFP4Quantizer()
        tensor = torch.randn(256, 256)

        print(f"\n原始张量: {tensor.shape}, dtype={tensor.dtype}")

        q_result = quantizer.quantize(tensor, return_dict=True)
        print(f"量化完成: 压缩比 {q_result['compression_ratio']:.2f}x")

        dq_tensor = quantizer.dequantize(
            q_result['quantized'],
            q_result['scales'],
            q_result['original_shape']
        )

        mse = torch.nn.functional.mse_loss(tensor, dq_tensor)
        print(f"反量化 MSE: {mse:.6f}")

        # 测试 nn.Linear 量化
        print("\n测试 MXFP4Linear:")
        linear = nn.Linear(768, 768)
        mxfp4_linear = MXFP4Linear.from_float(linear)

        x = torch.randn(8, 32, 768)
        with torch.no_grad():
            out1 = linear(x)
            out2 = mxfp4_linear(x)

        mse = torch.nn.functional.mse_loss(out1, out2)
        print(f"  输出 MSE: {mse:.6f}")

        print("✅ MXFP4 量化测试通过")
        return True

    except Exception as e:
        logger.error(f"MXFP4 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moe_optimized():
    """测试 GPU 优化 MoE"""
    print("\n" + "=" * 70)
    print("测试 2: GPU 优化 MoE")
    print("=" * 70)

    try:
        from apt_model.modeling.moe_optimized import (
            MoELayerOptimized,
            MoEConfig,
            create_moe_layer
        )

        config = MoEConfig(
            num_experts=8,
            top_k=2,
            expert_hidden_dim=2048
        )

        print(f"\n配置: {config.num_experts} 专家, top-{config.top_k}")

        moe = create_moe_layer(768, config)
        print(f"MoE 层参数量: {sum(p.numel() for p in moe.parameters()):,}")

        # 前向传播
        hidden_states = torch.randn(4, 128, 768)
        output, aux = moe(hidden_states, return_aux=True)

        print(f"\n输入: {hidden_states.shape}")
        print(f"输出: {output.shape}")
        print(f"辅助损失: {aux['aux_loss']:.6f}")
        print(f"负载均衡: {aux['balance_loss']:.6f}")
        print(f"路由熵: {aux['router_entropy']:.4f}")

        print("\n专家使用率:")
        for i, usage in enumerate(aux['expert_usage']):
            print(f"  专家 {i}: {usage:.2%}")

        print("✅ GPU 优化 MoE 测试通过")
        return True

    except Exception as e:
        logger.error(f"MoE 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extreme_scale_training():
    """测试超大规模训练配置"""
    print("\n" + "=" * 70)
    print("测试 3: 超大规模训练配置")
    print("=" * 70)

    try:
        from apt_model.optimization.extreme_scale_training import (
            ExtremeScaleConfig,
            CommunicationTopology,
            ParallelismManager
        )

        # 测试配置
        config = ExtremeScaleConfig(
            total_gpus=100000,
            data_parallel_size=128,
            tensor_parallel_size=8,
            pipeline_parallel_size=8
        )

        print(f"\n配置:")
        print(f"  总 GPU 数: {config.total_gpus:,}")
        print(f"  Data Parallel: {config.data_parallel_size}")
        print(f"  Tensor Parallel: {config.tensor_parallel_size}")
        print(f"  Pipeline Parallel: {config.pipeline_parallel_size}")

        # 测试通信拓扑
        topology = CommunicationTopology(config)
        print(f"\n通信拓扑:")
        print(f"  GPUs per rack: {topology.gpus_per_rack}")
        print(f"  GPUs per datacenter: {topology.gpus_per_datacenter}")

        # 测试通信成本
        cost_intra = topology.get_communication_cost(0, 10)
        cost_inter = topology.get_communication_cost(0, 100)
        cost_dc = topology.get_communication_cost(0, 60000)

        print(f"\n通信成本（相对延迟）:")
        print(f"  Intra-rack: {cost_intra:.1f}x")
        print(f"  Inter-rack: {cost_inter:.1f}x")
        print(f"  Inter-datacenter: {cost_dc:.1f}x")

        print("✅ 超大规模训练配置测试通过")
        return True

    except Exception as e:
        logger.error(f"超大规模训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_virtual_blackwell_integration():
    """测试 Virtual Blackwell 集成"""
    print("\n" + "=" * 70)
    print("测试 4: Virtual Blackwell 集成")
    print("=" * 70)

    try:
        import apt_model.optimization.vb_global as vb

        # 测试基础启用
        print("\n测试基础启用:")
        vb.enable(
            use_mxfp4=True,
            use_flash_attn=True,
            mixed_precision=True,
            verbose=False
        )

        assert vb.is_enabled(), "VB应该已启用"
        config = vb.get_config()
        assert config['use_mxfp4'], "MXFP4应该已启用"

        print("✓ 基础启用正常")

        # 测试禁用
        vb.disable()
        assert not vb.is_enabled(), "VB应该已禁用"
        print("✓ 禁用正常")

        # 测试 MoE 模式
        print("\n测试 MoE 模式:")
        vb.enable_moe_mode(num_experts=8, top_k=2)
        config = vb.get_config()
        assert config['use_moe_optimized'], "MoE应该已启用"
        print("✓ MoE模式正常")

        # 测试超大规模模式
        print("\n测试超大规模模式:")
        vb.enable_extreme_scale_mode(total_gpus=100000)
        config = vb.get_config()
        assert config['enable_extreme_scale'], "超大规模训练应该已启用"
        print("✓ 超大规模模式正常")

        print("\n✅ Virtual Blackwell 集成测试通过")
        return True

    except Exception as e:
        logger.error(f"Virtual Blackwell 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_workflow():
    """测试完整集成工作流"""
    print("\n" + "=" * 70)
    print("测试 5: 完整集成工作流")
    print("=" * 70)

    try:
        import apt_model.optimization.vb_global as vb
        from apt_model.optimization.mxfp4_quantization import MXFP4Linear

        # 1. 创建简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 512)
                self.fc2 = nn.Linear(512, 256)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleModel()
        print(f"\n原始模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 2. 启用 VB（全优化）
        vb.enable_full_optimization()

        # 3. 应用 MXFP4 量化
        quantized_model = vb.apply_mxfp4_to_model(model)
        print(f"量化后模型: {type(quantized_model)}")

        # 4. 测试前向传播
        x = torch.randn(8, 256)
        with torch.no_grad():
            out1 = model(x)
            out2 = quantized_model(x)

        mse = torch.nn.functional.mse_loss(out1, out2)
        print(f"输出 MSE: {mse:.6f}")

        # 5. 获取统计
        stats = vb.get_stats()
        print(f"\n统计信息: {stats}")

        print("\n✅ 完整集成工作流测试通过")
        return True

    except Exception as e:
        logger.error(f"集成工作流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("🚀 极限优化测试套件")
    print("=" * 70)
    print("测试内容:")
    print("  1. MXFP4 量化")
    print("  2. GPU 优化 MoE")
    print("  3. 超大规模训练配置")
    print("  4. Virtual Blackwell 集成")
    print("  5. 完整集成工作流")
    print("=" * 70)

    results = []

    # 运行所有测试
    tests = [
        ("MXFP4 量化", test_mxfp4_quantization),
        ("GPU 优化 MoE", test_moe_optimized),
        ("超大规模训练", test_extreme_scale_training),
        ("Virtual Blackwell", test_virtual_blackwell_integration),
        ("完整工作流", test_integration_workflow),
    ]

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"测试 '{name}' 崩溃: {e}")
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} {status}")

    print("=" * 70)
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
