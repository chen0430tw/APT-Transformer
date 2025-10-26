#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型管理员模式启动脚本
快速启动APT管理员模式的便捷脚本
"""

import sys
import os
import argparse
from pathlib import Path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="APT模型管理员模式启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置启动
  python start_admin_mode.py
  
  # 指定模型路径
  python start_admin_mode.py --model-path /path/to/model
  
  # 使用自定义密码
  python start_admin_mode.py --password mypassword123
  
  # 使用中文字符级分词器
  python start_admin_mode.py --tokenizer-type chinese-char
  
  # 强制使用CPU（当CUDA出错时）
  python start_admin_mode.py --force-cpu
        """
    )
    
    # 基本参数
    parser.add_argument('--model-path', type=str, default="apt_model", 
                       help="模型路径 (默认: apt_model)")
    parser.add_argument('--password', type=str, default="aptadmin", 
                       help="管理员密码 (默认: aptadmin)")
    
    # 生成参数
    parser.add_argument('--temperature', type=float, default=0.7, 
                       help="生成温度 (默认: 0.7)")
    parser.add_argument('--top-p', type=float, default=0.9, 
                       help="Top-p采样参数 (默认: 0.9)")
    parser.add_argument('--max-length', type=int, default=100, 
                       help="最大生成长度 (默认: 100)")
    
    # 分词器选项
    parser.add_argument('--tokenizer-type', type=str, 
                       choices=['gpt2', 'chinese-char', 'chinese-word'],
                       help="指定分词器类型")
    
    # 设备选项
    parser.add_argument('--force-cpu', action='store_true', 
                       help="强制使用CPU（避免CUDA错误）")
    
    # 调试选项
    parser.add_argument('--verbose', action='store_true',
                       help="显示详细日志")
    
    args = parser.parse_args()
    
    # 显示启动信息
    print("=" * 70)
    print("🚀 APT模型管理员模式启动器")
    print("=" * 70)
    print(f"📁 模型路径: {args.model_path}")
    print(f"🌡️  温度: {args.temperature}")
    print(f"📊 Top-p: {args.top_p}")
    print(f"📏 最大长度: {args.max_length}")
    print(f"💻 设备: {'CPU (强制)' if args.force_cpu else 'Auto (GPU/CPU)'}")
    if args.tokenizer_type:
        print(f"🔤 分词器: {args.tokenizer_type}")
    print("=" * 70)
    print()
    
    try:
        # 尝试导入APT管理员模式模块
        try:
            # 方法1: 如果已安装为包
            from apt_model.interactive.admin_mode import start_admin_mode
            print("✅ 从已安装的包加载管理员模式")
        except ImportError:
            # 方法2: 从当前目录加载
            current_dir = Path(__file__).parent
            admin_mode_path = current_dir / 'admin_mode.py'
            
            if admin_mode_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("admin_mode", admin_mode_path)
                admin_mode_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(admin_mode_module)
                start_admin_mode = admin_mode_module.start_admin_mode
                print("✅ 从当前目录加载管理员模式")
            else:
                raise ImportError("无法找到admin_mode.py文件")
        
        print()
        
        # 启动管理员模式
        start_admin_mode(
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            max_length=args.max_length,
            admin_password=args.password,
            tokenizer_type=args.tokenizer_type,
            force_cpu=args.force_cpu
        )
        
    except ImportError as e:
        print("\n❌ 错误: 无法导入APT管理员模式模块")
        print("\n可能的原因:")
        print("1. APT模型包未正确安装")
        print("2. admin_mode.py文件不在正确位置")
        print("\n解决方案:")
        print("方案1 - 如果使用已安装的包:")
        print("  1. 确保在APT模型项目根目录")
        print("  2. 运行: pip install -e .")
        print("  3. 确保admin_mode.py在apt_model/interactive/目录下")
        print("\n方案2 - 如果直接运行脚本:")
        print("  1. 确保admin_mode.py在同一目录")
        print("  2. 运行: python start_admin_mode.py")
        print(f"\n详细错误: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ 启动管理员模式时出错: {e}")
        
        if args.verbose:
            import traceback
            print("\n详细错误信息:")
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
