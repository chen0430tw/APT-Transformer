import sys
import torch

# ========== 天眼系统：追踪 BFloat16 mask 错误 ==========
# 这会极其消耗性能，仅用于调试！
torch.autograd.set_detect_anomaly(True)
print("[ANOMALY DETECTION] 已启用！这会极其消耗性能，仅用于调试！")

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
