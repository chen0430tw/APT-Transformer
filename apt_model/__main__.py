#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apt_model - Legacy CLI entry point

⚠️ This is a compatibility layer. New projects should use APT 2.0.

Usage:
    python -m apt_model chat        # Interactive chat
    python -m apt_model train       # Training (redirects to quickstart)
    python -m apt_model --help      # Show help

For APT 2.0 (recommended):
    python quickstart.py            # New recommended way
"""

import sys
import warnings


def show_help():
    """Show help message"""
    print("""
APT Model - Legacy CLI (Compatibility Layer)

⚠️  WARNING: This is APT 1.0 compatibility mode
⚠️  Recommended: Use APT 2.0 instead (see below)

Available commands:
    python -m apt_model chat        Interactive chat
    python -m apt_model train       Start training
    python -m apt_model --help      Show this help

APT 2.0 (Recommended):
    python quickstart.py --help                    # Quick start
    python quickstart.py --profile lite --demo     # View config
    python quickstart.py --profile lite            # Start training

    Or use Python API:
        from apt.core.config import load_profile
        from apt.trainops.engine import Trainer

        config = load_profile('lite')
        trainer = Trainer(config)
        trainer.train()

Documentation:
    - APT 2.0 Guide: docs/ARCHITECTURE_2.0.md
    - Migration Guide: docs/ARCHITECTURE_2.0.md
    - All docs: docs/README.md

Why migrate to APT 2.0?
    ✅ Profile-based configuration (YAML)
    ✅ Virtual Blackwell support (100K+ GPUs)
    ✅ Better architecture (DDD)
    ✅ All new features only in APT 2.0
""")


def run_chat():
    """Run interactive chat"""
    print("🤖 Starting APT Chat (Compatibility Mode)...")
    print("⚠️  Note: This uses APT 1.0 compatibility layer")
    print()

    # Check if we have a trained model
    import os
    if not os.path.exists('checkpoints'):
        print("❌ Error: No trained model found")
        print("   Please train a model first:")
        print("   python quickstart.py --profile lite")
        print()
        return 1

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
    """Run training (redirect to quickstart)"""
    print("🚂 APT Training (Compatibility Mode)")
    print()
    print("⚠️  Redirecting to APT 2.0 quickstart...")
    print()
    print("For better experience, use:")
    print("    python quickstart.py --help")
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
    # Show deprecation warning
    warnings.warn(
        "apt_model CLI is deprecated. Please use APT 2.0 (quickstart.py or apt.* package)",
        DeprecationWarning,
        stacklevel=2
    )

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
