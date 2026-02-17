#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟Blackwell GPU性能测试脚本
Virtual Blackwell GPU Benchmark Tool

功能：
- 自动检测硬件配置（GPU、CPU、内存、SSD）
- 测试不同规模模型的支持能力
- 评估虚拟GPU堆叠性能
- 生成详细测试报告

用法：
    python examples/benchmarks/vblackwell_benchmark.py
    python examples/benchmarks/vblackwell_benchmark.py --quick
    python examples/benchmarks/vblackwell_benchmark.py --detailed
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print("=" * 80)
print("🎮 虚拟Blackwell GPU性能测试工具")
print("=" * 80)
print()


class HardwareDetector:
    """硬件检测器"""

    def __init__(self):
        self.info = {
            'gpu': [],
            'cpu': {},
            'memory': {},
            'storage': {}
        }

    def detect_all(self) -> Dict:
        """检测所有硬件"""
        print("🔍 检测硬件配置...")
        print()

        self._detect_gpu()
        self._detect_cpu()
        self._detect_memory()
        self._detect_storage()

        return self.info

    def _detect_gpu(self):
        """检测GPU"""
        print("📊 GPU检测:")

        # 尝试检测CUDA GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"  ✓ 检测到 {gpu_count} 张 NVIDIA GPU")

                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        'id': i,
                        'name': props.name,
                        'total_memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}",
                        'type': 'cuda'
                    }
                    self.info['gpu'].append(gpu_info)
                    print(f"    GPU {i}: {props.name}")
                    print(f"      - 显存: {gpu_info['total_memory_gb']:.1f} GB")
                    print(f"      - 算力: {gpu_info['compute_capability']}")
            else:
                print("  ⚠️  未检测到CUDA GPU")
        except ImportError:
            print("  ⚠️  PyTorch未安装，跳过CUDA检测")

        # 尝试检测NPU（华为昇腾）
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                npu_count = torch_npu.npu.device_count()
                print(f"  ✓ 检测到 {npu_count} 张华为昇腾NPU")
                for i in range(npu_count):
                    self.info['gpu'].append({
                        'id': i,
                        'name': 'Huawei Ascend NPU',
                        'type': 'npu'
                    })
        except ImportError:
            pass

        # 如果没有GPU
        if not self.info['gpu']:
            print("  ℹ️  CPU-only模式")
            self.info['gpu'].append({
                'id': 0,
                'name': 'CPU Only',
                'type': 'cpu'
            })

        print()

    def _detect_cpu(self):
        """检测CPU"""
        print("🖥️  CPU检测:")

        import platform
        import multiprocessing

        self.info['cpu'] = {
            'model': platform.processor() or 'Unknown',
            'cores': multiprocessing.cpu_count(),
            'architecture': platform.machine()
        }

        print(f"  CPU型号: {self.info['cpu']['model']}")
        print(f"  核心数: {self.info['cpu']['cores']}")
        print(f"  架构: {self.info['cpu']['architecture']}")
        print()

    def _detect_memory(self):
        """检测内存"""
        print("💾 内存检测:")

        try:
            import psutil
            mem = psutil.virtual_memory()
            self.info['memory'] = {
                'total_gb': mem.total / (1024**3),
                'available_gb': mem.available / (1024**3),
                'used_percent': mem.percent
            }
            print(f"  总内存: {self.info['memory']['total_gb']:.1f} GB")
            print(f"  可用内存: {self.info['memory']['available_gb']:.1f} GB")
            print(f"  使用率: {self.info['memory']['used_percent']:.1f}%")
        except ImportError:
            print("  ⚠️  psutil未安装，使用估算值")
            self.info['memory'] = {
                'total_gb': 16.0,  # 默认估算
                'available_gb': 8.0,
                'used_percent': 50.0
            }
            print("  总内存: ~16 GB (估算)")

        print()

    def _detect_storage(self):
        """检测存储"""
        print("💿 存储检测:")

        try:
            import psutil
            disk = psutil.disk_usage('/')
            self.info['storage'] = {
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'used_percent': disk.percent
            }
            print(f"  总容量: {self.info['storage']['total_gb']:.1f} GB")
            print(f"  可用空间: {self.info['storage']['free_gb']:.1f} GB")
            print(f"  使用率: {self.info['storage']['used_percent']:.1f}%")
        except ImportError:
            print("  ⚠️  psutil未安装，跳过存储检测")
            self.info['storage'] = {
                'total_gb': 500.0,
                'free_gb': 250.0,
                'used_percent': 50.0
            }

        print()


class VirtualBlackwellBenchmark:
    """虚拟Blackwell性能测试"""

    def __init__(self, hardware_info: Dict):
        self.hardware_info = hardware_info
        self.results = []

    def run_tests(self, test_mode: str = 'standard'):
        """运行测试"""
        print("=" * 80)
        print("🧪 开始虚拟Blackwell性能测试")
        print("=" * 80)
        print()

        if test_mode == 'quick':
            test_configs = self._get_quick_tests()
        elif test_mode == 'detailed':
            test_configs = self._get_detailed_tests()
        else:
            test_configs = self._get_standard_tests()

        for i, config in enumerate(test_configs, 1):
            print(f"[测试 {i}/{len(test_configs)}] {config['name']}")
            result = self._run_single_test(config)
            self.results.append(result)
            print()

        return self.results

    def _get_quick_tests(self) -> List[Dict]:
        """快速测试配置"""
        return [
            {'name': '小型模型 (7B)', 'model_size_gb': 14, 'num_gpus': 1},
            {'name': '中型模型 (13B)', 'model_size_gb': 26, 'num_gpus': 1},
        ]

    def _get_standard_tests(self) -> List[Dict]:
        """标准测试配置"""
        return [
            {'name': '小型模型 (7B, FP16)', 'model_size_gb': 14, 'num_gpus': 1},
            {'name': '小型模型 (7B, MXFP4)', 'model_size_gb': 3.5, 'num_gpus': 1},
            {'name': '中型模型 (13B, FP16)', 'model_size_gb': 26, 'num_gpus': 1},
            {'name': '大型模型 (70B, FP16)', 'model_size_gb': 140, 'num_gpus': 1},
            {'name': '大型模型 (70B, MXFP4)', 'model_size_gb': 35, 'num_gpus': 1},
        ]

    def _get_detailed_tests(self) -> List[Dict]:
        """详细测试配置"""
        return [
            {'name': '小型模型 (7B, FP16)', 'model_size_gb': 14, 'num_gpus': 1},
            {'name': '小型模型 (7B, MXFP4)', 'model_size_gb': 3.5, 'num_gpus': 1},
            {'name': '中型模型 (13B, FP16)', 'model_size_gb': 26, 'num_gpus': 1},
            {'name': '中型模型 (13B, MXFP4)', 'model_size_gb': 6.5, 'num_gpus': 1},
            {'name': '大型模型 (70B, FP16)', 'model_size_gb': 140, 'num_gpus': 1},
            {'name': '大型模型 (70B, MXFP4)', 'model_size_gb': 35, 'num_gpus': 1},
            {'name': '超大模型 (175B, FP16)', 'model_size_gb': 350, 'num_gpus': 4},
            {'name': '超大模型 (405B, MXFP4)', 'model_size_gb': 101, 'num_gpus': 4},
        ]

    def _run_single_test(self, config: Dict) -> Dict:
        """运行单个测试"""
        model_size = config['model_size_gb']
        num_gpus = config['num_gpus']

        # 计算虚拟Blackwell配置
        vb_config = self._calculate_vb_usage(model_size, num_gpus)

        # 检查是否可行
        feasibility = self._check_feasibility(vb_config, num_gpus)

        # 估算性能
        performance = self._estimate_performance(vb_config, feasibility)

        result = {
            'config': config,
            'vb_config': vb_config,
            'feasibility': feasibility,
            'performance': performance
        }

        self._print_result(result)

        return result

    def _calculate_vb_usage(self, model_size_gb: float, num_gpus: int) -> Dict:
        """计算虚拟Blackwell使用量"""
        # 默认单GPU配置
        gpu_cache_gb = 2.0
        cpu_cache_gb = 8.0
        ssd_cache_gb = 32.0

        # 如果有实际GPU，调整GPU缓存大小
        if self.hardware_info['gpu'] and self.hardware_info['gpu'][0]['type'] == 'cuda':
            actual_gpu_mem = self.hardware_info['gpu'][0].get('total_memory_gb', 2.0)
            gpu_cache_gb = min(actual_gpu_mem * 0.8, model_size_gb)  # 使用80%显存

        # 分配到各层级
        per_gpu_size = model_size_gb / num_gpus
        remaining = per_gpu_size

        gpu_actual = min(remaining, gpu_cache_gb)
        remaining -= gpu_actual

        cpu_actual = min(remaining, cpu_cache_gb)
        remaining -= cpu_actual

        ssd_actual = min(remaining, ssd_cache_gb)
        remaining -= ssd_actual

        return {
            'gpu_gb': gpu_actual * num_gpus,
            'cpu_gb': cpu_actual * num_gpus,
            'ssd_gb': ssd_actual * num_gpus,
            'overflow_gb': remaining * num_gpus,
            'total_gb': model_size_gb,
            'num_gpus': num_gpus
        }

    def _check_feasibility(self, vb_config: Dict, num_gpus: int) -> Dict:
        """检查可行性"""
        hw = self.hardware_info

        # 检查GPU
        gpu_ok = len(hw['gpu']) >= num_gpus
        if gpu_ok and hw['gpu'][0]['type'] == 'cuda':
            gpu_mem_ok = hw['gpu'][0].get('total_memory_gb', 0) >= vb_config['gpu_gb'] / num_gpus
        else:
            gpu_mem_ok = True  # CPU模式

        # 检查CPU内存
        cpu_ok = hw['memory']['total_gb'] >= vb_config['cpu_gb'] + 4  # +4GB系统开销

        # 检查SSD空间
        ssd_ok = hw['storage']['free_gb'] >= vb_config['ssd_gb'] + vb_config['total_gb']

        # 检查溢出
        overflow_ok = vb_config['overflow_gb'] < 0.1  # 容忍小于100MB溢出

        overall_ok = gpu_ok and gpu_mem_ok and cpu_ok and ssd_ok and overflow_ok

        return {
            'overall': overall_ok,
            'gpu_available': gpu_ok,
            'gpu_memory': gpu_mem_ok,
            'cpu_memory': cpu_ok,
            'storage': ssd_ok,
            'no_overflow': overflow_ok
        }

    def _estimate_performance(self, vb_config: Dict, feasibility: Dict) -> Dict:
        """估算性能"""
        if not feasibility['overall']:
            return {
                'score': 0,
                'rating': 'N/A',
                'bottleneck': self._identify_bottleneck(feasibility)
            }

        # 计算分数（0-100）
        gpu_ratio = vb_config['gpu_gb'] / vb_config['total_gb'] if vb_config['total_gb'] > 0 else 0
        cpu_ratio = vb_config['cpu_gb'] / vb_config['total_gb'] if vb_config['total_gb'] > 0 else 0
        ssd_ratio = vb_config['ssd_gb'] / vb_config['total_gb'] if vb_config['total_gb'] > 0 else 0

        # GPU数据最快，权重最高
        score = (
            gpu_ratio * 90 +  # GPU缓存命中率最高
            cpu_ratio * 50 +  # CPU中速
            ssd_ratio * 10    # SSD最慢
        )

        # 评级
        if score >= 80:
            rating = '优秀 (Excellent)'
        elif score >= 60:
            rating = '良好 (Good)'
        elif score >= 40:
            rating = '一般 (Fair)'
        elif score >= 20:
            rating = '较差 (Poor)'
        else:
            rating = '不推荐 (Not Recommended)'

        return {
            'score': round(score, 1),
            'rating': rating,
            'gpu_hit_rate': round(gpu_ratio * 100, 1),
            'cpu_hit_rate': round(cpu_ratio * 100, 1),
            'ssd_hit_rate': round(ssd_ratio * 100, 1)
        }

    def _identify_bottleneck(self, feasibility: Dict) -> str:
        """识别瓶颈"""
        if not feasibility['gpu_available']:
            return 'GPU数量不足'
        if not feasibility['gpu_memory']:
            return 'GPU显存不足'
        if not feasibility['cpu_memory']:
            return 'CPU内存不足'
        if not feasibility['storage']:
            return 'SSD空间不足'
        if not feasibility['no_overflow']:
            return '模型大小超出虚拟容量上限'
        return '未知'

    def _print_result(self, result: Dict):
        """打印测试结果"""
        config = result['config']
        vb = result['vb_config']
        feas = result['feasibility']
        perf = result['performance']

        print(f"  模型大小: {config['model_size_gb']:.1f} GB")
        print(f"  使用GPU: {vb['num_gpus']}张")
        print()
        print(f"  虚拟Blackwell分配:")
        print(f"    GPU缓存: {vb['gpu_gb']:.2f} GB")
        print(f"    CPU缓存: {vb['cpu_gb']:.2f} GB")
        print(f"    SSD缓存: {vb['ssd_gb']:.2f} GB")
        if vb['overflow_gb'] > 0.1:
            print(f"    ⚠️  溢出: {vb['overflow_gb']:.2f} GB")
        print()

        if feas['overall']:
            print(f"  ✅ 可行性: 通过")
            print(f"  📊 性能评分: {perf['score']}/100 - {perf['rating']}")
            print(f"  🎯 GPU命中率: {perf['gpu_hit_rate']:.1f}%")
        else:
            print(f"  ❌ 可行性: 不通过")
            print(f"  🚫 瓶颈: {perf.get('bottleneck', '未知')}")


class ReportGenerator:
    """报告生成器"""

    def __init__(self, hardware_info: Dict, results: List[Dict]):
        self.hardware_info = hardware_info
        self.results = results

    def generate(self):
        """生成报告"""
        print("=" * 80)
        print("📄 测试报告")
        print("=" * 80)
        print()

        self._print_summary()
        self._print_recommendations()
        self._save_json_report()

    def _print_summary(self):
        """打印汇总"""
        print("📊 测试汇总:")
        print()

        total = len(self.results)
        passed = sum(1 for r in self.results if r['feasibility']['overall'])

        print(f"  总测试数: {total}")
        print(f"  通过: {passed}")
        print(f"  失败: {total - passed}")
        print(f"  通过率: {passed/total*100:.1f}%")
        print()

        # 最佳配置
        feasible_results = [r for r in self.results if r['feasibility']['overall']]
        if feasible_results:
            best = max(feasible_results, key=lambda r: r['performance']['score'])
            print(f"  🏆 最佳配置: {best['config']['name']}")
            print(f"     性能评分: {best['performance']['score']}/100")
            print(f"     模型大小: {best['config']['model_size_gb']:.1f} GB")
            print()

    def _print_recommendations(self):
        """打印建议"""
        print("💡 建议:")
        print()

        hw = self.hardware_info

        # GPU建议
        if hw['gpu'] and hw['gpu'][0]['type'] == 'cuda':
            gpu_mem = hw['gpu'][0].get('total_memory_gb', 0)
            if gpu_mem < 8:
                print("  ⚠️  GPU显存较小，建议:")
                print("     - 使用MXFP4量化减少75%内存占用")
                print("     - 训练小型模型 (7B-13B)")
            elif gpu_mem < 24:
                print("  ✓ GPU显存中等，适合:")
                print("     - 训练中型模型 (13B-30B)")
                print("     - 使用MXFP4量化训练70B模型")
            else:
                print("  ✓ GPU显存充足，适合:")
                print("     - 训练大型模型 (70B+)")
                print("     - 或使用多GPU训练超大模型")
        else:
            print("  ℹ️  CPU-only模式:")
            print("     - 推荐使用小型量化模型 (7B MXFP4)")
            print("     - 考虑使用云GPU服务")

        print()

        # CPU内存建议
        cpu_mem = hw['memory']['total_gb']
        if cpu_mem < 16:
            print("  ⚠️  CPU内存较小 (<16GB):")
            print("     - 建议升级到32GB")
        elif cpu_mem < 64:
            print("  ✓ CPU内存适中 (16-64GB)")
        else:
            print("  ✓ CPU内存充足 (>=64GB)")

        print()

        # 存储建议
        storage_free = hw['storage']['free_gb']
        if storage_free < 100:
            print("  ⚠️  SSD可用空间不足 (<100GB):")
            print("     - 建议清理磁盘空间")
            print("     - 或添加额外存储")
        else:
            print("  ✓ 存储空间充足")

        print()

    def _save_json_report(self):
        """保存JSON报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'hardware': self.hardware_info,
            'results': self.results
        }

        output_file = f"vblackwell_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(os.path.dirname(__file__), output_file)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📁 详细报告已保存: {output_file}")
        except Exception as e:
            print(f"⚠️  保存报告失败: {e}")

        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='虚拟Blackwell GPU性能测试工具')
    parser.add_argument('--quick', action='store_true', help='快速测试（仅2个配置）')
    parser.add_argument('--detailed', action='store_true', help='详细测试（8个配置）')
    args = parser.parse_args()

    # 确定测试模式
    if args.quick:
        test_mode = 'quick'
    elif args.detailed:
        test_mode = 'detailed'
    else:
        test_mode = 'standard'

    print(f"测试模式: {test_mode.upper()}")
    print()

    # 1. 硬件检测
    detector = HardwareDetector()
    hardware_info = detector.detect_all()

    # 2. 运行测试
    benchmark = VirtualBlackwellBenchmark(hardware_info)
    results = benchmark.run_tests(test_mode)

    # 3. 生成报告
    report_gen = ReportGenerator(hardware_info, results)
    report_gen.generate()

    print("=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
