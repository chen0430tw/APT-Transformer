#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APX (Model Packaging) CLI Commands

Provides CLI commands for APX model packaging and management:
- pack-apx: Package model into APX format
- detect-capabilities: Auto-detect model capabilities
- detect-framework: Detect model framework
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from apt.apps.tools.apx import (
    pack_apx,
    detect_framework,
    detect_capabilities,
    APXPackagingError,
)
from apt.apps.cli.command_registry import register_command

logger = logging.getLogger(__name__)


def run_pack_apx_command(args):
    """
    Package a model into APX format.

    Args:
        args: Command line arguments with:
            - src: Source model directory
            - out: Output APX file path
            - name: Model name
            - version: Model version
            - adapter: Adapter type (hf or stub)
            - mode: Packaging mode (full or thin)
            - weights_glob: Optional weight file pattern
            - tokenizer_glob: Optional tokenizer file pattern
            - config_file: Optional explicit config file
            - prefers: Preference (builtin or plugin)
            - capability: List of explicit capabilities
            - compose: List of compose key=value items
            - add_test: Whether to add smoke test
            - no_auto_detect: Disable auto capability detection

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print(f"[APX] Packaging model: {args.src} -> {args.out}")

    try:
        src_repo = Path(args.src)
        out_apx = Path(args.out)

        # Validate inputs
        if not src_repo.exists():
            print(f"[error] Source directory not found: {src_repo}")
            return 1

        # Detect framework
        framework = detect_framework(src_repo)
        print(f"[info] Detected framework: {framework}")

        # Package model
        pack_apx(
            src_repo=src_repo,
            out_apx=out_apx,
            name=args.name,
            version=args.version,
            adapter=args.adapter,
            mode=args.mode,
            weights_glob=args.weights_glob if hasattr(args, 'weights_glob') else None,
            tokenizer_glob=args.tokenizer_glob if hasattr(args, 'tokenizer_glob') else None,
            config_file=args.config_file if hasattr(args, 'config_file') else None,
            prefers=args.prefers if hasattr(args, 'prefers') else "builtin",
            capabilities=args.capability if hasattr(args, 'capability') else None,
            compose_items=args.compose if hasattr(args, 'compose') else None,
            add_test=args.add_test if hasattr(args, 'add_test') else True,
            auto_detect_capabilities=not (hasattr(args, 'no_auto_detect') and args.no_auto_detect),
        )

        print(f"[success] APX package created successfully!")
        return 0

    except APXPackagingError as e:
        print(f"[error] APX packaging failed: {e}")
        logger.error(f"APX packaging error: {e}")
        return 1
    except Exception as e:
        print(f"[error] Unexpected error: {e}")
        logger.error(f"Unexpected error in pack-apx: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def run_detect_capabilities_command(args):
    """
    Auto-detect model capabilities.

    Args:
        args: Command line arguments with:
            - src: Source model directory

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print(f"[APX] Detecting capabilities: {args.src}")

    try:
        src_repo = Path(args.src)

        if not src_repo.exists():
            print(f"[error] Source directory not found: {src_repo}")
            return 1

        # Detect capabilities
        capabilities = detect_capabilities(src_repo)

        if capabilities:
            print(f"[info] Detected capabilities:")
            for cap in capabilities:
                print(f"  - {cap}")
        else:
            print(f"[info] No special capabilities detected")

        return 0

    except Exception as e:
        print(f"[error] Detection failed: {e}")
        logger.error(f"Capability detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def run_detect_framework_command(args):
    """
    Detect model framework.

    Args:
        args: Command line arguments with:
            - src: Source model directory

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print(f"[APX] Detecting framework: {args.src}")

    try:
        src_repo = Path(args.src)

        if not src_repo.exists():
            print(f"[error] Source directory not found: {src_repo}")
            return 1

        # Detect framework
        framework = detect_framework(src_repo)
        print(f"[info] Detected framework: {framework}")

        return 0

    except Exception as e:
        print(f"[error] Detection failed: {e}")
        logger.error(f"Framework detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def run_apx_info_command(args):
    """
    Display information about an APX package.

    Args:
        args: Command line arguments with:
            - apx: Path to APX file

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print(f"[APX] Reading package info: {args.apx}")

    try:
        import zipfile
        import yaml

        apx_path = Path(args.apx)

        if not apx_path.exists():
            print(f"[error] APX file not found: {apx_path}")
            return 1

        # Read apx.yaml from ZIP
        with zipfile.ZipFile(apx_path, 'r') as zf:
            if 'apx.yaml' not in zf.namelist():
                print(f"[error] Invalid APX file: apx.yaml not found")
                return 1

            with zf.open('apx.yaml') as f:
                content = f.read().decode('utf-8')
                # Parse YAML manually (avoid pyyaml dependency)
                print(f"\n[APX Manifest]")
                print(content)

        return 0

    except Exception as e:
        print(f"[error] Failed to read APX info: {e}")
        logger.error(f"APX info error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


# Register APX commands
def register_apx_commands():
    """
    Register all APX commands with the command registry.
    """
    register_command(
        name="pack-apx",
        func=run_pack_apx_command,
        category="tools",
        help_text="Package a model into APX format",
        args_help={
            "src": "Source model directory",
            "out": "Output APX file path",
            "name": "Model name",
            "version": "Model version",
            "adapter": "Adapter type (hf or stub, default: hf)",
            "mode": "Packaging mode (full or thin, default: full)",
            "weights-glob": "Weight file glob pattern (optional)",
            "tokenizer-glob": "Tokenizer file glob pattern (optional)",
            "config-file": "Explicit config file path (optional)",
            "prefers": "Preference: builtin or plugin (default: builtin)",
            "capability": "Explicit capability (can be specified multiple times)",
            "compose": "Compose key=value (can be specified multiple times)",
            "add-test": "Add smoke test (default: true)",
            "no-auto-detect": "Disable auto capability detection",
        },
        aliases=["apx-pack", "package"],
    )

    register_command(
        name="detect-capabilities",
        func=run_detect_capabilities_command,
        category="tools",
        help_text="Auto-detect model capabilities (MoE, RAG, RLHF, etc.)",
        args_help={
            "src": "Source model directory",
        },
        aliases=["detect-caps", "caps"],
    )

    register_command(
        name="detect-framework",
        func=run_detect_framework_command,
        category="tools",
        help_text="Detect model framework (huggingface, structured, etc.)",
        args_help={
            "src": "Source model directory",
        },
        aliases=["detect-fw", "framework"],
    )

    register_command(
        name="apx-info",
        func=run_apx_info_command,
        category="tools",
        help_text="Display information about an APX package",
        args_help={
            "apx": "Path to APX file",
        },
        aliases=["info-apx"],
    )

    logger.debug("Registered APX commands: pack-apx, detect-capabilities, detect-framework, apx-info")
