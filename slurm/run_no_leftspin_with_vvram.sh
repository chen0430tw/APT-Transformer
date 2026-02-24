#!/bin/bash
#SBATCH --job-name=apt_no_leftspin_vvram
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=no_leftspin_vvram_%j.out
#SBATCH --error=no_leftspin_vvram_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT 测试 - 禁用LeftSpin，保留Virtual VRAM"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "开始时间: $(date)"
echo "============================================"

# 创建临时Python脚本禁用LeftSpin（放在工作目录以解决导入问题）
cat > disable_leftspin_wrapper.py << 'EOF'
import sys
import torch

# 调试：检查Virtual VRAM是否可用
try:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram
    print("[DEBUG] Virtual VRAM import: OK")
    print(f"[DEBUG] virtual_vram module: {virtual_vram}")
    print(f"[DEBUG] VirtualVRAMConfig: {VirtualVRAMConfig}")
except Exception as e:
    print(f"[DEBUG] Virtual VRAM import FAILED: {e}")
    virtual_vram = None
    VirtualVRAMConfig = None

# Hook模型创建过程
original_init = None

def disable_leftspin_init(self, *args, **kwargs):
    result = original_init(self, *args, **kwargs)
    # 禁用LeftSpin
    if hasattr(self, 'use_left_spin'):
        self.use_left_spin = False
    if hasattr(self, 'left_spin_attn'):
        self.left_spin_attn = None
    if hasattr(self, 'left_spin_ffn'):
        self.left_spin_ffn = None
    print("[DISABLE LEFTSPIN] LeftSpin components disabled")
    return result

# 在模型创建前hook
import apt.model.architectures.apt_model as apt_model_module
original_init = apt_model_module.APTModel.__init__
apt_model_module.APTModel.__init__ = disable_leftspin_init

# 导入并运行训练
if __name__ == "__main__":
    import argparse
    from apt.trainops.scripts import pretrain_quickcook
    pretrain_quickcook.main()
EOF

# 使用wrapper运行训练
srun python disable_leftspin_wrapper.py \
    --output-dir ./test_no_leftspin_vvram \
    --max-steps 10 \
    --save-interval 10 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --no-c4 \
    --no-mc4 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --use-virtual-vram

echo "============================================"
echo "测试完成时间: $(date)"
echo "============================================"
