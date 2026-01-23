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
╔══════════════════════════════════════════════════════════════╗
║          APT Model - Command Line Interface                  ║
╚══════════════════════════════════════════════════════════════╝

📋 Basic Commands:
    python -m apt_model chat        Interactive chat with trained model
    python -m apt_model train       Start training
    python -m apt_model --help      Show this help

🌐 Web Services:
    python -m apt_model.webui.app --checkpoint-dir ./checkpoints
        Launch WebUI (Gradio interface)
        Options: --port, --share, --username, --password

    python -m apt_model.api.server --checkpoint-dir ./checkpoints
        Launch REST API server (FastAPI)
        Options: --port, --host, --api-key, --reload

📊 Training Options:
    python -m apt_model train [options]
        --epochs N          Number of training epochs (default: 10)
        --batch-size N      Batch size (default: 16)
        --data PATH         Training data path

💬 Chat Options:
    python -m apt_model chat
        Requires trained model in ./checkpoints/
        Supports: *.pt, *.pth, *.bin files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 Alternative Entry Points:

1. Quick Start CLI (Profile-based)
    python quickstart.py --help
    python quickstart.py --profile lite --demo
    python quickstart.py --profile lite

2. Python API (Programmatic)
    from apt.core.config import load_profile
    from apt.trainops.engine import Trainer

    config = load_profile('lite')
    trainer = Trainer(config)
    trainer.train()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 Documentation:
    • CLI Guide:         docs/CLI_STATUS.md
    • APT 2.0 Architecture: docs/ARCHITECTURE_2.0.md
    • All Documentation: docs/README.md

💡 Choose your preferred entry point:
    • apt_model CLI    - Traditional command-line interface (this)
    • quickstart.py    - Profile-based quick start
    • apt.* API        - Programmatic Python API

Examples:
    # Start WebUI on custom port
    python -m apt_model.webui.app --port 8080 --checkpoint-dir ./my_models

    # Train with custom parameters
    python -m apt_model train --epochs 50 --batch-size 32 --data ./data.txt

    # Launch API with authentication
    python -m apt_model.api.server --api-key my-secret-key --port 8000

    # Interactive chat
    python -m apt_model chat
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
