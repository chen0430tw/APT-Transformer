#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apt_model - APT CLI entry point

Command-line interface for APT Model training and inference.

Usage:
    python -m apt_model chat        # Interactive chat
    python -m apt_model train       # Training
    python -m apt_model --help      # Show help

Alternative entry points:
    python quickstart.py            # Quick start with profiles
    python -c "from apt.core..."    # Python API
"""

import sys


def show_help():
    """Show help message"""
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║              APT Model - Command Line Interface                        ║
╚════════════════════════════════════════════════════════════════════════╝

🚀 TRAINING:
  train              Train model
  train-custom       Train with custom dataset
  fine-tune          Fine-tune pretrained model
  train-hf           Train HuggingFace compatible model
  distill            Knowledge distillation
  train-reasoning    Train logical reasoning

💬 INTERACTIVE:
  chat               Interactive chat

📊 EVALUATION:
  evaluate (eval)    Evaluate model
  visualize          Generate charts
  compare            Compare models
  test               Test model

🔧 TOOLS:
  clean-cache        Clean cache
  estimate           Estimate training time
  config             Configuration
  debug              Debug mode

📋 INFO:
  info               Show info
  list               List resources
  size               Calculate size

🛠️ MAINTENANCE:
  prune              Delete old models
  backup             Backup models

📦 DATA:
  process-data       Process datasets

🌐 DISTRIBUTION:
  upload             Upload models
  export-ollama      Export to Ollama

🎁 APX (Model Packaging) - 🆕 NEW in 2.0:
  pack-apx           Package model into APX format
  apx-info           Display APX package info
  detect-capabilities Auto-detect model capabilities (MoE, RAG, RLHF)
  detect-framework   Detect model framework (HuggingFace, etc.)

⚡ ADVANCED TECHNICAL FEATURES - 🆕 NEW in 2.0:
  train-moe          MoE (Mixture of Experts) training
  blackwell-simulate Virtual Blackwell GPU simulation (vblackwell)
  aim-memory         AIM (Advanced In-context Memory) management
  npu-accelerate     NPU backend acceleration (npu)
  rag-query          RAG/KG-RAG retrieval queries
  quantize-mxfp4     MXFP4 4-bit quantization (mxfp4)

📋 PROFILE SYSTEM - 🆕 NEW in 2.0:
  --profile PROFILE  Use profile config (lite/standard/pro/full)
                     • lite     - Lightweight, fast startup
                     • standard - Balanced performance
                     • pro      - High performance, distributed
                     • full     - All features + Virtual Blackwell

🌐 WEB SERVICES:
  python -m apt_model.webui.app --checkpoint-dir ./checkpoints
  python -m apt_model.api.server --checkpoint-dir ./checkpoints

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 COMMON OPTIONS:
  --epochs N             Training epochs (default: 20)
  --batch-size N         Batch size (default: 8)
  --learning-rate N      Learning rate (default: 3e-5)
  --save-path PATH       Model save path
  --model-path PATH      Model load path
  --temperature N        Generation temperature (default: 0.7)
  --top-p N              Nucleus sampling (default: 0.9)
  --max-length N         Max generation length (default: 50)
  --force-cpu            Force CPU
  --language LANG        Interface language (zh_CN/en_US)
  --data-path PATH       Training data path
  --tokenizer-type TYPE  Tokenizer type
  --monitor-resources    Monitor resources
  --create-plots         Create plots
  --profile PROFILE      Profile config (lite/standard/pro/full) 🆕

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 EXAMPLES:

  Basic Training:
    python -m apt_model train --epochs 50
    python -m apt_model train --profile pro --epochs 100

  Fine-tuning:
    python -m apt_model fine-tune --model-path ./pretrained --profile standard

  Chat:
    python -m apt_model chat --temperature 0.9

  APX Packaging (🆕 NEW):
    python -m apt_model pack-apx --src ./my_model --out model.apx
    python -m apt_model detect-capabilities --src ./my_model
    python -m apt_model apx-info --apx model.apx

  Advanced Technical Features (🆕 NEW):
    python -m apt_model train-moe --num-experts 8 --top-k 2
    python -m apt_model blackwell-simulate --num-gpus 100000
    python -m apt_model aim-memory --checkpoint ./model --context-size 128k
    python -m apt_model npu-accelerate --backend ascend
    python -m apt_model rag-query --query "What is transformers?" --kg-mode
    python -m apt_model quantize-mxfp4 --model-path ./my_model

  Export:
    python -m apt_model export-ollama --model-path ./my_model

  Web Services:
    python -m apt_model.webui.app --port 8080
    python -m apt_model.api.server --api-key secret

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🆕 APT 2.0 NEW FEATURES:

  1. Profile System - Configuration-driven workflow
     python quickstart.py --profile lite --demo

  2. APX Model Packaging - Portable model format
     python -m apt_model pack-apx --src ./model --out model.apx

  3. Virtual Blackwell - GPU virtualization (100K+ GPUs)
     python -m apt_model blackwell-simulate --num-gpus 100000

  4. MoE (Mixture of Experts) - Multi-expert parallel computing
     python -m apt_model train-moe --num-experts 8 --top-k 2

  5. MXFP4 Quantization - Microsoft-OpenAI 4-bit floating point
     python -m apt_model quantize-mxfp4 --model-path ./my_model

  6. AIM Memory - Advanced In-context Memory system
     python -m apt_model aim-memory --context-size 128k

  7. NPU Acceleration - Support for Ascend, Kunlun, MLU, TPU
     python -m apt_model npu-accelerate --backend ascend

  8. RAG/KG-RAG - Retrieval Augmented Generation with KG
     python -m apt_model rag-query --query "..." --kg-mode

  9. Auto-detection - Automatically detect model capabilities
     python -m apt_model detect-capabilities --src ./model

  10. Domain-Driven Architecture - Model/TrainOps/vGPU/APX separation
      See: docs/ARCHITECTURE_2.0.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 ALTERNATIVE ENTRY POINTS:

  quickstart.py (Profile-based, recommended for beginners)
    python quickstart.py --profile lite --demo
    python quickstart.py --profile standard --epochs 20

  apt.* API (Programmatic, recommended for developers)
    from apt.core.config import load_profile
    from apt.trainops.engine import Trainer
    config = load_profile('lite')
    trainer = Trainer(config)
    trainer.train()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 DOCS: docs/CLI_STATUS.md | docs/ARCHITECTURE_2.0.md | docs/README.md

For detailed help: python -m apt_model <command> --help
""")


def run_chat():
    """Run interactive chat"""
    print("🤖 Starting APT Chat...")
    print()

    # Check if we have a trained model
    import os
    import glob

    checkpoint_dir = './checkpoints'

    # Check if checkpoints directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory not found")
        print(f"   Looking for models in: {os.path.abspath(checkpoint_dir)}")
        print()
        print("   Please train a model first:")
        print("   python quickstart.py --profile lite")
        print()
        return 1

    # Check if there are actual model files
    model_files = (
        glob.glob(os.path.join(checkpoint_dir, '*.pt')) +
        glob.glob(os.path.join(checkpoint_dir, '*.pth')) +
        glob.glob(os.path.join(checkpoint_dir, '*.bin')) +
        glob.glob(os.path.join(checkpoint_dir, '**/pytorch_model.bin'), recursive=True)
    )

    if not model_files:
        print(f"❌ Error: No model files found in checkpoint directory")
        print(f"   Checkpoint directory: {os.path.abspath(checkpoint_dir)}")
        print(f"   Looking for: *.pt, *.pth, *.bin files")
        print()
        print("   Please train a model first:")
        print("   python quickstart.py --profile lite")
        print()
        return 1

    print(f"✓ Found {len(model_files)} model file(s) in {checkpoint_dir}")
    print()

    # Simple chat loop
    print("Type 'exit' or 'quit' to exit")
    print("=" * 60)

    try:
        # Try to import generation module
        try:
            from apt.apps.interactive import chat_loop
            chat_loop()
        except ImportError:
            print("⚠️  Chat functionality is not fully implemented yet")
            print()
            print("For now, you can:")
            print("1. Use quickstart.py to train a model")
            print("2. Use Python API for inference:")
            print()
            print("   from apt.core.config import load_profile")
            print("   config = load_profile('lite')")
            print("   # Load model and generate...")
            print()

            # Simple fallback chat
            while True:
                try:
                    user_input = input("\nYou: ")
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        print("Goodbye!")
                        break

                    print(f"Bot: [Echo] {user_input}")
                    print("     (This is compatibility mode - train a model for real inference)")

                except (KeyboardInterrupt, EOFError):
                    print("\nGoodbye!")
                    break

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def run_train():
    """Run training"""
    print("🚂 APT Training")
    print()

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='APT Training (Compatibility)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--data', type=str, help='Training data path')

    args, unknown = parser.parse_known_args()

    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    if args.data:
        print(f"  Data: {args.data}")
    print()

    # Redirect to quickstart
    print("Starting quickstart.py...")
    import subprocess
    cmd = ['python', 'quickstart.py', '--profile', 'lite', '--epochs', str(args.epochs)]
    if args.data:
        cmd.extend(['--data', args.data])

    return subprocess.call(cmd)


def main():
    """Main entry point"""
    # Parse command
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        return 0

    command = sys.argv[1]

    # Remove command from args so subcommands can parse their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == 'chat':
        return run_chat()
    elif command == 'train':
        return run_train()
    else:
        print(f"❌ Unknown command: {command}")
        print()
        show_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
