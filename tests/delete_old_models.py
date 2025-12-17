import os
import glob

# 删除旧模型
saved_models_dir = 'tests/saved_models'
if os.path.exists(saved_models_dir):
    old_models = glob.glob(os.path.join(saved_models_dir, '*.pt'))
    for model_path in old_models:
        print(f"删除旧模型: {model_path}")
        os.remove(model_path)
    print(f"\n✅ 已删除 {len(old_models)} 个旧模型")
    print("现在重新训练会使用新的tokenizer（包含语言标签）")
else:
    print("saved_models目录不存在")
