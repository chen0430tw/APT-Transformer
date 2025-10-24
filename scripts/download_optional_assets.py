#!/usr/bin/env python3
"""Utility script to download optional runtime dependencies.

The APT training pipeline can take advantage of scikit-learn based metrics and
GPT-2 tokenizers from Hugging Face.  These components are optional for basic
usage but recommended for full functionality.  This helper script installs the
Python packages when missing and caches the GPT-2 tokenizer files locally so
offline runs remain deterministic.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "apt_model" / "resources" / "gpt2_tokenizer"


def ensure_package(module_name: str, pip_target: str) -> None:
    """Install *pip_target* if *module_name* cannot be imported."""

    if importlib.util.find_spec(module_name) is not None:
        print(f"✅ {module_name} already available")
        return

    print(f"⬇️ Installing {pip_target} because {module_name} is missing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pip_target])


def download_gpt2_tokenizer(target_dir: Path, force: bool = False) -> Path:
    """Download the GPT-2 tokenizer files into *target_dir*.

    The function avoids re-downloading if the directory already contains the
    expected files unless *force* is True.
    """

    expected_files = {"tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"}
    if target_dir.exists():
        existing = {path.name for path in target_dir.iterdir() if path.is_file()}
        if expected_files.issubset(existing) and not force:
            print(f"✅ GPT-2 tokenizer already cached in {target_dir}")
            return target_dir
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    ensure_package("transformers", "transformers")

    # Import after ensuring the package is available.
    from transformers import GPT2TokenizerFast

    print("⬇️ Downloading GPT-2 tokenizer (gpt2)...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(target_dir)
    print(f"✅ GPT-2 tokenizer saved to {target_dir}")
    return target_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download optional APT assets")
    parser.add_argument(
        "--skip-sklearn",
        action="store_true",
        help="Do not attempt to install scikit-learn",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Do not download the GPT-2 tokenizer",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Target directory for cached GPT-2 tokenizer files",
    )
    parser.add_argument(
        "--force-tokenizer",
        action="store_true",
        help="Re-download the tokenizer even if files already exist",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.skip_sklearn:
        ensure_package("sklearn", "scikit-learn")
    else:
        print("⏭️ Skipping scikit-learn installation as requested")

    if not args.skip_tokenizer:
        download_dir = download_gpt2_tokenizer(args.tokenizer_dir, force=args.force_tokenizer)
        os.environ.setdefault("APT_GPT2_TOKENIZER_DIR", str(download_dir))
        print(
            "💡 Set the APT_GPT2_TOKENIZER_DIR environment variable to reuse the cached files",
            f"(export APT_GPT2_TOKENIZER_DIR={download_dir})",
        )
    else:
        print("⏭️ Skipping GPT-2 tokenizer download as requested")

    print("All requested assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
