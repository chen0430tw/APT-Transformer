#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APX Converter Core

Handles packaging of models into APX format.
"""

import os
import json
import glob
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from apt.apps.tools.apx.templates import (
    make_apx_yaml,
    write_text,
)
from apt.apps.tools.apx.adapters import get_adapter_code, AdapterType
from apt.apps.tools.apx.detectors import detect_capabilities


class APXPackagingError(Exception):
    """Exception raised during APX packaging"""
    pass


# Tokenizer file candidates
TOKENIZER_CANDIDATES = [
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "sp.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

# Weight file patterns
WEIGHT_GLOBS_DEFAULT = [
    "*.safetensors",
    "pytorch_model*.bin",
    "consolidated*.pth",
]


def find_first(root: Path, names: List[str]) -> Optional[Path]:
    """
    Find first existing file from a list of candidates.

    Args:
        root: Root directory to search in
        names: List of file names to search for

    Returns:
        Path to first found file, or None
    """
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def find_any_globs(root: Path, patterns: List[str]) -> List[Path]:
    """
    Find files matching any of the given glob patterns.

    Args:
        root: Root directory to search in
        patterns: List of glob patterns

    Returns:
        List of unique matching paths
    """
    results: List[Path] = []
    for pat in patterns:
        results.extend([Path(p) for p in glob.glob(str(root / pat))])

    # Remove duplicates
    uniq = []
    seen = set()
    for p in results:
        resolved = p.resolve()
        if resolved not in seen:
            uniq.append(p)
            seen.add(resolved)
    return uniq


def detect_framework(src: Path) -> str:
    """
    Detect model framework from directory structure.

    Args:
        src: Source model directory

    Returns:
        Framework name: "huggingface", "structured", or "unknown"
    """
    framework = "unknown"

    # Check for HuggingFace structure
    config_path = src / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                if "architectures" in config or "model_type" in config:
                    framework = "huggingface"
        except Exception:
            pass

    # Check for other structured formats
    other_configs = ["params.json", "lit_config.json", "config.yml", "model-index.json"]
    if any((src / n).exists() for n in other_configs):
        if framework == "unknown":
            framework = "structured"

    return framework


def collect_artifacts(
    src_repo: Path,
    mode: str,
    tmp_root: Path,
    weights_glob: Optional[str] = None,
    tokenizer_glob: Optional[str] = None,
    config_file: Optional[str] = None,
) -> Dict[str, str]:
    """
    Collect and copy model artifacts.

    Args:
        src_repo: Source model directory
        mode: Packaging mode ("full" or "thin")
        tmp_root: Temporary build directory
        weights_glob: Optional custom weight file glob pattern
        tokenizer_glob: Optional custom tokenizer file glob pattern
        config_file: Optional explicit config file path

    Returns:
        Dictionary mapping artifact names to paths

    Raises:
        APXPackagingError: If config file not found
    """
    artifacts_map: Dict[str, str] = {}

    # 1. Config file
    cfg_path = Path(config_file) if config_file else (src_repo / "config.json")
    if not cfg_path.exists():
        # Try to find config.json
        candidates = list(src_repo.glob("**/config.json"))
        cfg_path = candidates[0] if candidates else None

    if not cfg_path or not cfg_path.exists():
        raise APXPackagingError(
            "config.json not found; please specify config_file parameter"
        )

    # Copy or create placeholder
    if mode == "full":
        shutil.copy2(cfg_path, tmp_root / "artifacts/config.json")
    else:
        # Thin mode: placeholder with source path
        content = {"__thin__": True, "source_config": str(cfg_path.resolve())}
        write_text(
            tmp_root / "artifacts/config.json",
            json.dumps(content, ensure_ascii=False, indent=2),
        )
    artifacts_map["config"] = "artifacts/config.json"

    # 2. Tokenizer files
    tok_files = []
    if tokenizer_glob:
        tok_files = [Path(p) for p in glob.glob(str(src_repo / tokenizer_glob))]
    else:
        for n in TOKENIZER_CANDIDATES:
            p = src_repo / n
            if p.exists():
                tok_files.append(p)

    # Remove duplicates
    tok_files = list(dict.fromkeys([p.resolve() for p in tok_files]))

    had_tok = False
    for p in tok_files:
        had_tok = True
        if mode == "full":
            shutil.copy2(p, tmp_root / "artifacts" / p.name)

    if not had_tok:
        # Placeholder
        write_text(
            tmp_root / "artifacts/tokenizer.json",
            json.dumps(
                {"__thin__": True, "note": "no tokenizer files found"},
                ensure_ascii=False,
                indent=2,
            ),
        )
        artifacts_map["tokenizer"] = "artifacts/tokenizer.json"
    else:
        # Prefer tokenizer.json
        tok_json = next(
            (p for p in tok_files if p.name == "tokenizer.json"),
            tok_files[0],
        )
        artifacts_map["tokenizer"] = f"artifacts/{tok_json.name}"

    # 3. Weight files
    w_globs = [weights_glob] if weights_glob else WEIGHT_GLOBS_DEFAULT
    weight_files = find_any_globs(src_repo, w_globs)

    if not weight_files:
        print("[warn] No weight files matched; continuing with thin mode placeholders")

    if weight_files:
        chosen = weight_files[0]
        if mode == "full":
            shutil.copy2(chosen, tmp_root / "artifacts" / chosen.name)
            artifacts_map["weights"] = f"artifacts/{chosen.name}"
        else:
            write_text(
                tmp_root / "artifacts/weights.info",
                json.dumps(
                    {"__thin__": True, "source_weight": str(chosen.resolve())},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            artifacts_map["weights"] = "artifacts/weights.info"
    else:
        write_text(
            tmp_root / "artifacts/weights.info",
            json.dumps(
                {"__thin__": True, "note": "no weights matched"},
                ensure_ascii=False,
                indent=2,
            ),
        )
        artifacts_map["weights"] = "artifacts/weights.info"

    return artifacts_map


def pack_apx(
    src_repo: Path,
    out_apx: Path,
    name: str,
    version: str,
    adapter: str = "hf",
    mode: str = "full",
    weights_glob: Optional[str] = None,
    tokenizer_glob: Optional[str] = None,
    config_file: Optional[str] = None,
    prefers: str = "builtin",
    capabilities: Optional[List[str]] = None,
    compose_items: Optional[List[str]] = None,
    add_test: bool = True,
    auto_detect_capabilities: bool = True,
) -> None:
    """
    Package a model into APX format.

    Args:
        src_repo: Source model directory
        out_apx: Output APX file path
        name: Model name
        version: Model version
        adapter: Adapter type ("hf" or "stub")
        mode: Packaging mode ("full" or "thin")
        weights_glob: Optional custom weight file glob pattern
        tokenizer_glob: Optional custom tokenizer file glob pattern
        config_file: Optional explicit config file path
        prefers: Preference for builtin vs plugin ("builtin" or "plugin")
        capabilities: Optional explicit list of capabilities
        compose_items: Optional compose key=value items
        add_test: Whether to add smoke test
        auto_detect_capabilities: Whether to auto-detect capabilities

    Raises:
        APXPackagingError: If packaging fails
    """
    # Validate inputs
    src_repo = Path(src_repo).resolve()
    out_apx = Path(out_apx).resolve()

    if not src_repo.exists():
        raise APXPackagingError(f"Source directory not found: {src_repo}")

    # Create temporary build directory
    tmp_root = Path(".apx_build_tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    (tmp_root / "model/adapters").mkdir(parents=True, exist_ok=True)
    (tmp_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_root / "tests").mkdir(parents=True, exist_ok=True)

    try:
        # 1. Collect artifacts
        artifacts_map = collect_artifacts(
            src_repo=src_repo,
            mode=mode,
            tmp_root=tmp_root,
            weights_glob=weights_glob,
            tokenizer_glob=tokenizer_glob,
            config_file=config_file,
        )

        # 2. Generate adapters
        adapter_type = AdapterType.HF if adapter == "hf" else AdapterType.STUB
        adapter_code = get_adapter_code(adapter_type)

        if adapter == "hf":
            write_text(tmp_root / "model/adapters/hf_adapter.py", adapter_code["model"])
            write_text(
                tmp_root / "model/adapters/tokenizer_adapter.py",
                adapter_code["tokenizer"],
            )
            entry_model = "model/adapters/hf_adapter.py:HFAdapter"
            entry_tok = "model/adapters/tokenizer_adapter.py:HFTokenizerAdapter"
        else:
            write_text(
                tmp_root / "model/adapters/model_adapter.py", adapter_code["model"]
            )
            write_text(
                tmp_root / "model/adapters/tokenizer_adapter.py",
                adapter_code["tokenizer"],
            )
            entry_model = "model/adapters/model_adapter.py:DemoAdapter"
            entry_tok = "model/adapters/tokenizer_adapter.py:HFTokenizerAdapter"

        # 3. Auto-detect capabilities
        detected_caps = []
        if auto_detect_capabilities:
            detected_caps = detect_capabilities(src_repo)
            print(f"[info] Auto-detected capabilities: {detected_caps}")

        # Merge with explicit capabilities
        final_caps = list(set((capabilities or []) + detected_caps))

        # 4. Parse compose items
        compose = {}
        if compose_items:
            for kv in compose_items:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    compose[k.strip()] = v.strip()

        # 5. Generate apx.yaml
        yaml_txt = make_apx_yaml(
            name=name,
            version=version,
            entry_model=entry_model,
            entry_tokenizer=entry_tok,
            artifacts=artifacts_map,
            prefers=prefers,
            capabilities=final_caps,
            compose_kv=compose,
        )
        write_text(tmp_root / "apx.yaml", yaml_txt)

        # 6. Add smoke test
        if add_test:
            write_text(tmp_root / "tests/smoke.py", "print('smoke ok')\n")

        # 7. Package as ZIP
        out_apx.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_apx, "w", zipfile.ZIP_DEFLATED) as zf:
            for dp, _, fns in os.walk(tmp_root):
                for fn in fns:
                    full = Path(dp) / fn
                    rel = full.relative_to(tmp_root)
                    zf.write(str(full), arcname=str(rel))

        print(f"[ok] APX package created: {out_apx}")
        print(f"     Name: {name} v{version}")
        print(f"     Mode: {mode}")
        print(f"     Capabilities: {final_caps}")

    finally:
        # Cleanup
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
