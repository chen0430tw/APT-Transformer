#!/usr/bin/env python3
"""
HLBD Hardcore V2 训练启动器
自动检查环境并启动训练
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """检查所需依赖"""
    print("🔍 检查依赖...")

    missing = []

    # Check torch
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
        else:
            print("  ⚠️  CUDA not available (will use CPU)")
    except ImportError:
        missing.append("torch")
        print("  ❌ PyTorch not found")

    # Check numpy
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError:
        missing.append("numpy")
        print("  ❌ NumPy not found")

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\n安装命令:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  pip install numpy")
        return False

    print("✓ All dependencies OK\n")
    return True


def check_dataset():
    """检查数据集是否存在"""
    dataset_path = Path("data/HLBD_Hardcore_Full_V2.json")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\n生成数据集:")
        print("  python tools/generate_hlbd_hardcore_v2.py")
        return False

    size_mb = dataset_path.stat().st_size / 1024 / 1024
    print(f"✓ Dataset found: {dataset_path} ({size_mb:.2f} MB)")

    # Check sample count
    try:
        import json
        with open(dataset_path) as f:
            data = json.load(f)
            total = data.get("metadata", {}).get("total_samples", 0)
            print(f"  Samples: {total}")
    except Exception as e:
        print(f"  Warning: Could not read metadata: {e}")

    print()
    return True


def launch_training():
    """启动训练"""
    print("=" * 60)
    print("HLBD Hardcore V2 训练启动")
    print("数据集: 5042 samples | 数据稀释学 | 防模式坍缩")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check dataset
    if not check_dataset():
        return 1

    # Create output directories
    output_dir = Path("models/hlbd_hardcore_v2")
    log_dir = Path("logs/hlbd_hardcore_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output: {output_dir}")
    print(f"📁 Logs: {log_dir}")
    print()

    # Training parameters
    params = {
        "dataset": "data/HLBD_Hardcore_Full_V2.json",
        "output-dir": str(output_dir),
        "epochs": 50,
        "batch-size": 32,
        "learning-rate": 5e-5,
        "warmup-steps": 100,
        "save-every": 5,
        "eval-every": 5,
        "use-amp": True,
        "gradient-accumulation-steps": 2,
    }

    print("训练配置:")
    for key, value in params.items():
        if key == "use-amp" and value is True:
            print(f"  {key}: enabled")
        else:
            print(f"  {key}: {value}")
    print()

    # Build command
    cmd = ["python3", "training/train_hlbd_playground.py"]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    # Launch
    print("🚀 启动训练...")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("=" * 60)
    print()

    try:
        # Run training
        result = subprocess.run(cmd, check=True)

        print()
        print("=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        print(f"模型保存: {output_dir}")
        print(f"日志保存: {log_dir}")

        return result.returncode

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ 训练失败")
        print("=" * 60)
        return e.returncode

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠️  训练被用户中断")
        print("=" * 60)
        return 130


def main():
    """主函数"""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Launch training
    exit_code = launch_training()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
