"""
虚拟Blackwell自动补丁

自动拦截APT模型创建，透明应用VGPU优化。

使用方法：
    # 方法1：显式调用
    import apt_model.optimization.vb_autopatch as vb_patch
    vb_patch.patch_apt_models()

    # 方法2：导入即生效
    import apt_model.optimization.vb_autopatch  # 自动patch

    # 之后所有APTModel/APTLargeModel都会自动优化
    from apt.core.modeling.apt_model import APTLargeModel
    model = APTLargeModel(config)  # 已经是VGPU优化版本
"""

import sys
from typing import Optional
import torch.nn as nn

# 全局状态
_patched = False
_original_classes = {}


def patch_apt_models(verbose: bool = True):
    """
    Patch APT模型类，使其自动应用VGPU优化

    Args:
        verbose: 是否打印patch信息
    """
    global _patched, _original_classes

    if _patched:
        if verbose:
            print("⚠️ APT模型已经被patch过")
        return

    try:
        # 导入原始类
        from apt.core.modeling.apt_model import APTModel, APTLargeModel
        from apt_model.optimization import vb_global

        # 保存原始类
        _original_classes['APTModel'] = APTModel
        _original_classes['APTLargeModel'] = APTLargeModel

        # 创建包装类
        class VBAPTModel(APTModel):
            """自动应用VGPU优化的APTModel"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # 如果VB已启用，自动优化
                if vb_global.is_enabled():
                    self._apply_vb_optimization()

            def _apply_vb_optimization(self):
                """应用VGPU优化"""
                stack = vb_global.get_stack()
                if stack is None:
                    return

                # 替换Linear层为VGPUStackLinear
                from apt_model.optimization.vgpu_stack import VGPUStackLinear

                layer_count = 0
                for name, module in list(self.named_modules()):
                    if isinstance(module, nn.Linear):
                        # 获取父模块
                        parts = name.split('.')
                        if len(parts) == 1:
                            parent = self
                            attr_name = parts[0]
                        else:
                            parent = self
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            attr_name = parts[-1]

                        # 创建VGPU版本
                        vgpu_linear = VGPUStackLinear(
                            module.in_features,
                            module.out_features,
                            stack,
                            bias=module.bias is not None
                        )

                        # 复制权重
                        with torch.no_grad():
                            vgpu_linear.weight.copy_(module.weight)
                            if module.bias is not None:
                                vgpu_linear.bias.copy_(module.bias)

                        # 替换
                        setattr(parent, attr_name, vgpu_linear)
                        layer_count += 1

                config = vb_global.get_config()
                if config.get('verbose', True):
                    print(f"✓ 自动优化模型: {layer_count} 个线性层")

        class VBAPTLargeModel(APTLargeModel):
            """自动应用VGPU优化的APTLargeModel"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # 如果VB已启用，自动优化
                if vb_global.is_enabled():
                    self._apply_vb_optimization()

            def _apply_vb_optimization(self):
                """应用VGPU优化"""
                stack = vb_global.get_stack()
                if stack is None:
                    return

                # 替换Linear层为VGPUStackLinear
                from apt_model.optimization.vgpu_stack import VGPUStackLinear
                import torch

                layer_count = 0
                for name, module in list(self.named_modules()):
                    if isinstance(module, nn.Linear):
                        # 获取父模块
                        parts = name.split('.')
                        if len(parts) == 1:
                            parent = self
                            attr_name = parts[0]
                        else:
                            parent = self
                            for part in parts[:-1]:
                                parent = getattr(parent, part)
                            attr_name = parts[-1]

                        # 创建VGPU版本
                        vgpu_linear = VGPUStackLinear(
                            module.in_features,
                            module.out_features,
                            stack,
                            bias=module.bias is not None
                        )

                        # 复制权重
                        with torch.no_grad():
                            vgpu_linear.weight.copy_(module.weight)
                            if module.bias is not None:
                                vgpu_linear.bias.copy_(module.bias)

                        # 替换
                        setattr(parent, attr_name, vgpu_linear)
                        layer_count += 1

                config = vb_global.get_config()
                if config.get('verbose', True) and layer_count > 0:
                    print(f"✓ 自动优化模型: {layer_count} 个线性层")

        # 替换模块中的类
        import apt.core.modeling.apt_model as apt_module
        apt_module.APTModel = VBAPTModel
        apt_module.APTLargeModel = VBAPTLargeModel

        # 也替换sys.modules中的引用
        if 'apt_model.modeling.apt_model' in sys.modules:
            mod = sys.modules['apt_model.modeling.apt_model']
            mod.APTModel = VBAPTModel
            mod.APTLargeModel = VBAPTLargeModel

        _patched = True

        if verbose:
            print("\n" + "="*70)
            print("✅ APT模型已自动patch")
            print("="*70)
            print("所有新创建的APTModel/APTLargeModel将自动应用VGPU优化")
            print("="*70 + "\n")

    except Exception as e:
        print(f"❌ Patch失败: {e}")
        import traceback
        traceback.print_exc()


def unpatch_apt_models(verbose: bool = True):
    """恢复原始APT模型类"""
    global _patched, _original_classes

    if not _patched:
        if verbose:
            print("⚠️ APT模型未被patch")
        return

    try:
        # 恢复原始类
        import apt.core.modeling.apt_model as apt_module
        apt_module.APTModel = _original_classes['APTModel']
        apt_module.APTLargeModel = _original_classes['APTLargeModel']

        # 恢复sys.modules
        if 'apt_model.modeling.apt_model' in sys.modules:
            mod = sys.modules['apt_model.modeling.apt_model']
            mod.APTModel = _original_classes['APTModel']
            mod.APTLargeModel = _original_classes['APTLargeModel']

        _patched = False

        if verbose:
            print("✅ APT模型已恢复原始版本")

    except Exception as e:
        print(f"❌ Unpatch失败: {e}")


def is_patched() -> bool:
    """检查是否已patch"""
    return _patched


# 自动patch（可通过环境变量控制）
import os
if os.getenv('VB_AUTO_PATCH', '').lower() in ('1', 'true', 'yes'):
    patch_apt_models(verbose=True)


if __name__ == "__main__":
    # 测试
    print("虚拟Blackwell自动补丁")
    print("\n使用示例:")
    print("```python")
    print("# 方法1：手动patch")
    print("import apt_model.optimization.vb_autopatch as vb_patch")
    print("import apt_model.optimization.vb_global as vb")
    print("")
    print("vb.enable()  # 启用虚拟Blackwell")
    print("vb_patch.patch_apt_models()  # Patch APT模型")
    print("")
    print("# 方法2：环境变量自动patch")
    print("# export VB_AUTO_PATCH=1")
    print("# export ENABLE_VIRTUAL_BLACKWELL=1")
    print("")
    print("# 之后所有APT模型都会自动优化")
    print("from apt.core.modeling.apt_model import APTLargeModel")
    print("model = APTLargeModel(config)  # 已自动应用VGPU优化")
    print("```")
